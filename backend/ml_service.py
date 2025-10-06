import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
import time

from models import ExoplanetPredictionRequest, PredictionResult
from config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_COLUMNS_PATH, FEATURE_DEFAULTS, LABEL_ENCODER_PATH, IMPUTER_PATH

logger = logging.getLogger(__name__)

class ExoplanetPredictor:
    """
    Exoplanet prediction service using Random Forest model
    """
    
    def __init__(self):
        self.model: RandomForestClassifier = None
        self.scaler: StandardScaler = None
        self.label_encoder = None
        self.imputer = None
        self.feature_columns: List[str] = []
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained Random Forest model and all preprocessing components"""
        try:
            # Load Random Forest model
            if Path(MODEL_PATH).exists():
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"Random Forest model loaded from {MODEL_PATH}")
            else:
                logger.error(f"Model file not found: {MODEL_PATH}")
                return False
            
            # Load scaler
            if Path(PREPROCESSOR_PATH).exists():
                self.scaler = joblib.load(PREPROCESSOR_PATH)
                logger.info(f"Scaler loaded from {PREPROCESSOR_PATH}")
            else:
                logger.warning(f"Scaler file not found: {PREPROCESSOR_PATH}")
                self.scaler = StandardScaler()
            
            # Load label encoder
            if Path(LABEL_ENCODER_PATH).exists():
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                logger.info(f"Label encoder loaded from {LABEL_ENCODER_PATH}")
            else:
                logger.warning(f"Label encoder file not found: {LABEL_ENCODER_PATH}")
            
            # Load imputer
            if Path(IMPUTER_PATH).exists():
                self.imputer = joblib.load(IMPUTER_PATH)
                logger.info(f"Imputer loaded from {IMPUTER_PATH}")
            else:
                logger.warning(f"Imputer file not found: {IMPUTER_PATH}")
            
            # Load feature columns
            if Path(FEATURE_COLUMNS_PATH).exists():
                with open(FEATURE_COLUMNS_PATH, 'r') as f:
                    self.feature_columns = [line.strip() for line in f.readlines()]
                logger.info(f"Feature columns loaded: {len(self.feature_columns)} features")
            else:
                # Default feature columns based on your data
                self.feature_columns = [
                    'orbital_period_days', 'transit_duration_hours', 'transit_depth_ppm',
                    'planet_radius_re', 'equilibrium_temp_k', 'insolation_flux_earth',
                    'stellar_teff_k', 'stellar_radius_re', 'apparent_mag', 'ra', 'dec',
                    'mission_encoded'
                ]
                logger.warning("Using default feature columns")
            
            self.is_loaded = True
            logger.info("Exoplanet predictor loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_data(self, request: ExoplanetPredictionRequest) -> np.ndarray:
        """Preprocess input data for prediction"""
        try:
            # Convert request to dictionary
            data = request.dict()
            
            # Fill missing values with defaults
            for field, default_value in FEATURE_DEFAULTS.items():
                if data.get(field) is None:
                    data[field] = default_value
            
            # Encode mission (Kepler=0, TESS=1)
            mission_encoded = 1 if data['mission'].upper() == 'TESS' else 0
            data['mission_encoded'] = mission_encoded
            
            # Map input field names to model feature names
            feature_mapping = {
                'ORB_PERIOD': 'orbital_period_days',
                'transit_duration_hours': 'transit_duration_hours',
                'TRANSIT_DEPTH': 'transit_depth_ppm',
                'PLANET_RADIUS': 'planet_radius_re',
                'equilibrium_temp_k': 'equilibrium_temp_k',
                'INSOL_FLUX': 'insolation_flux_earth',
                'stellar_teff_k': 'stellar_teff_k',
                'STELLAR_RADIUS': 'stellar_radius_re',
                'apparent_mag': 'apparent_mag',
                'ra': 'ra',
                'dec': 'dec',
                'R_PLANET_R_STAR_RATIO': None,  # Calculated field
                'DEPTH_PER_RADIUS': None,  # Calculated field
                'mission_encoded': None  # Special handling
            }
            
            # Create feature vector in the correct order
            feature_vector = []
            for column in self.feature_columns:
                if column == 'mission_encoded':
                    feature_vector.append(mission_encoded)
                elif column == 'R_PLANET_R_STAR_RATIO':
                    # Calculate planet radius to stellar radius ratio
                    planet_radius = data.get('planet_radius_re', FEATURE_DEFAULTS.get('planet_radius_re', 1.0))
                    stellar_radius = data.get('stellar_radius_re', FEATURE_DEFAULTS.get('stellar_radius_re', 1.0))
                    ratio = planet_radius / stellar_radius if stellar_radius != 0 else 0
                    feature_vector.append(ratio)
                elif column == 'DEPTH_PER_RADIUS':
                    # Calculate transit depth per planet radius
                    depth = data.get('transit_depth_ppm', FEATURE_DEFAULTS.get('transit_depth_ppm', 1000.0))
                    planet_radius = data.get('planet_radius_re', FEATURE_DEFAULTS.get('planet_radius_re', 1.0))
                    depth_per_radius = depth / planet_radius if planet_radius != 0 else 0
                    feature_vector.append(depth_per_radius)
                else:
                    # Map to the correct input field name
                    input_field = feature_mapping.get(column, column)
                    if input_field:
                        feature_vector.append(data.get(input_field, FEATURE_DEFAULTS.get(input_field, 0.0)))
                    else:
                        feature_vector.append(0.0)
            
            # Convert to numpy array and reshape
            features = np.array(feature_vector).reshape(1, -1)
            
            # Apply preprocessing if available
            if self.scaler is not None and hasattr(self.scaler, 'transform'):
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def predict_single(self, request: ExoplanetPredictionRequest) -> PredictionResult:
        """Make a single prediction"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess input
            features = self.preprocess_data(request)
            logger.debug(f"Input features for prediction: {features}")
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            logger.info(f"Raw model.predict_proba output: {probability}")
            logger.info(f"Model classes: {getattr(self.model, 'classes_', 'Unknown')}")
            logger.info(f"Feature vector shape: {features.shape}")
            
            # Get probability for the positive class (exoplanet)
            exoplanet_probability = probability[1] if len(probability) > 1 else probability[0]
            non_exoplanet_probability = probability[0] if len(probability) > 1 else (1 - probability[0])
            
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Probability of NOT being a planet (class 0): {non_exoplanet_probability}")
            logger.info(f"Probability of being a planet (class 1): {exoplanet_probability}")
            logger.info(f"Sum of probabilities: {non_exoplanet_probability + exoplanet_probability}")
            

            
            # Determine confidence level and message
            confidence_level = self._get_confidence_level(exoplanet_probability, prediction)
            confidence_message = self._get_confidence_message(exoplanet_probability, prediction)
            
            # Calculate feature importance and z-scores for analysis
            feature_analysis = self._calculate_feature_analysis(features[0], request)
            feature_analysis["confidence_message"] = confidence_message
            
            # Add diagnostic information
            feature_analysis["model_diagnostic"] = {
                "raw_probabilities": probability.tolist(),
                "class_0_probability": float(probability[0]) if len(probability) > 1 else float(1 - probability[0]),
                "class_1_probability": float(probability[1]) if len(probability) > 1 else float(probability[0]),
                "model_classes": getattr(self.model, 'classes_', []).tolist()
            }
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            result = PredictionResult(
                prediction=int(prediction),
                probability=float(exoplanet_probability),
                confidence_level=confidence_level,
                processing_time_ms=processing_time
            )
            
            # Add feature analysis to result
            result.feature_analysis = feature_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _calculate_feature_analysis(self, features: np.ndarray, request: ExoplanetPredictionRequest) -> Dict[str, Any]:
        """Calculate feature importance and z-scores for visualization"""
        try:
            # Get feature importances from the model
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Create feature importance mapping
                feature_importance = dict(zip(self.feature_columns, importances))
                
                # Calculate z-scores (standardized feature values)
                # If we have a scaler, the features are already standardized
                z_scores = {}
                for i, column in enumerate(self.feature_columns):
                    if i < len(features):
                        z_scores[column] = float(features[i])
                
                # Get top features by importance
                top_features = []
                for column, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                    z_score = z_scores.get(column, 0.0)
                    top_features.append({
                        "name": column,
                        "importance": float(importance),
                        "z": float(z_score)
                    })
                
                return {
                    "feature_importance": feature_importance,
                    "z_scores": z_scores,
                    "top_features": top_features
                }
            else:
                return {"error": "Model does not have feature importances"}
                
        except Exception as e:
            logger.error(f"Error calculating feature analysis: {e}")
            return {"error": str(e)}
    
    def predict_batch(self, requests: List[ExoplanetPredictionRequest]) -> List[PredictionResult]:
        """Make batch predictions"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        try:
            # Preprocess all requests
            features_list = []
            for idx, request in enumerate(requests):
                features = self.preprocess_data(request)
                logger.debug(f"Batch input features [{idx}]: {features}")
                features_list.append(features[0])  # Remove the reshape dimension

            # Convert to numpy array
            all_features = np.array(features_list)
            logger.debug(f"All batch features shape: {all_features.shape}")
            # Check for all-constant or all-default features
            if np.all(all_features == all_features[0]):
                logger.warning("All batch input features are identical. This may cause uniform predictions.")

            # Make batch predictions
            start_time = time.time()
            predictions = self.model.predict(all_features)
            probabilities = self.model.predict_proba(all_features)
            logger.debug(f"Batch raw model.predict_proba output: {probabilities}")
            processing_time = (time.time() - start_time) * 1000

            # Process results
            for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
                logger.debug(f"Row {i} probability array: {probability}")
                exoplanet_probability = probability[1] if len(probability) > 1 else probability[0]
                logger.info(f"Batch Prediction [{i}]: {prediction}, Probability: {exoplanet_probability}")
                confidence_level = self._get_confidence_level(exoplanet_probability, prediction)

                results.append(PredictionResult(
                    prediction=int(prediction),
                    probability=float(exoplanet_probability),
                    confidence_level=confidence_level,
                    processing_time_ms=processing_time / len(requests)  # Average time per prediction
                ))

            return results

        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
            raise
    
    def _get_confidence_level(self, probability: float, prediction: int) -> str:
        """Determine confidence level based on probability and prediction"""
        # For exoplanet prediction (1): high confidence when probability is high
        # For non-exoplanet prediction (0): high confidence when probability is low
        if prediction == 1:  # Predicted as exoplanet
            if probability >= 0.8:
                return "High"
            elif probability >= 0.6:
                return "Medium"
            else:
                return "Low"
        else:  # Predicted as non-exoplanet
            if probability <= 0.2:
                return "High"
            elif probability <= 0.4:
                return "Medium"
            else:
                return "Low"
    
    def _get_confidence_message(self, probability: float, prediction: int) -> str:
        """Generate a human-readable confidence message"""
        if prediction == 1:  # Predicted as exoplanet
            # Add more nuanced messaging based on probability ranges
            if probability >= 0.95:
                return "Very likely planet"
            elif probability >= 0.8:
                return "Likely planet"
            elif probability >= 0.6:
                return "Possibly planet"
            else:
                return "Uncertain - leaning planet"
        else:  # Predicted as non-exoplanet (prediction == 0)
            # For non-exoplanet predictions, we want to show confidence in NOT being a planet
            # High confidence means low probability (close to 0)
            if probability <= 0.05:
                return "Very likely not a planet"
            elif probability <= 0.2:
                return "Likely not a planet"
            elif probability <= 0.4:
                return "Possibly not a planet"
            else:
                return "Uncertain - leaning not a planet"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        info = {
            "model_type": "RandomForestClassifier",
            "is_loaded": self.is_loaded,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns
        }
        
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            # Sort by importance
            info["feature_importance"] = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        return info

# Global predictor instance
predictor = ExoplanetPredictor()

def initialize_predictor() -> bool:
    """Initialize the global predictor instance"""
    return predictor.load_model()

def get_predictor() -> ExoplanetPredictor:
    """Get the global predictor instance"""
    return predictor
