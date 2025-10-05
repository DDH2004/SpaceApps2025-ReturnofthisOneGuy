import os
import numpy as np
import logging
from typing import List, Any
from pathlib import Path
import io

from models import ExoplanetPredictionRequest
from config import (
    FEATURE_DEFAULTS,
    TABULAR_TORCH_MODEL,
    ENHANCED_FUSION_MODEL,
    EXO_TABULAR_FEATURES,
)

import torch
from PIL import Image
import importlib.util

logger = logging.getLogger(__name__)


class TorchModelManager:
    """Manager for loading PyTorch models from the exoplanet pipeline and running inference."""

    def __init__(self):
        self.tabular_model = None
        self.fusion_model = None
        self.device = torch.device('cpu')
        self.tabular_input_size = EXO_TABULAR_FEATURES
        self.is_loaded = False

    def load_tabular_model(self, path: str = TABULAR_TORCH_MODEL) -> bool:
        try:
            if not Path(path).exists():
                logger.error(f"Tabular torch model not found: {path}")
                return False
            # Dynamically import the TabularNet class from the repo's src/models.py
            models_py = Path(path).parents[1] / 'src' / 'models.py'
            if not models_py.exists():
                logger.error(f"Cannot find models.py at expected location: {models_py}")
                return False

            spec = importlib.util.spec_from_file_location('exo_models', str(models_py))
            exo_models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exo_models)  # type: ignore

            TabularNet = getattr(exo_models, 'TabularNet')

            model = TabularNet(input_size=self.tabular_input_size)
            state = torch.load(path, map_location=self.device)
            # Allow both state_dict or full model
            if isinstance(state, dict):
                # state is a state_dict -> load into model
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    logger.error(f"Failed to load state_dict into TabularNet: {e}")
                    return False
            elif isinstance(state, torch.nn.Module):
                # saved full model object
                model = state
            else:
                logger.error(f"Unrecognized tabular model file contents: {type(state)}")
                return False

            model.to(self.device)
            model.eval()
            self.tabular_model = model
            # Try to infer the expected input size from the model's first Linear layer
            try:
                inferred_size = None
                for m in model.modules():
                    if isinstance(m, torch.nn.Linear):
                        inferred_size = int(m.in_features)
                        break
                if inferred_size is not None:
                    self.tabular_input_size = inferred_size
                    logger.info(f"Inferred tabular input size: {self.tabular_input_size}")
                else:
                    logger.info(f"Could not infer tabular input size; using default {self.tabular_input_size}")
            except Exception as e:
                logger.warning(f"Failed to infer tabular input size: {e}")
            self.is_loaded = True
            logger.info(f"Loaded tabular torch model from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load tabular torch model: {e}")
            return False

    def load_fusion_model(self, path: str = ENHANCED_FUSION_MODEL) -> bool:
        try:
            if not Path(path).exists():
                logger.error(f"Fusion model not found: {path}")
                return False
            # Dynamically import HybridEnsemble from src/models.py
            models_py = Path(path).parents[1] / 'src' / 'models.py'
            if not models_py.exists():
                logger.error(f"Cannot find models.py at expected location: {models_py}")
                return False

            spec = importlib.util.spec_from_file_location('exo_models', str(models_py))
            exo_models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(exo_models)  # type: ignore

            HybridEnsemble = getattr(exo_models, 'HybridEnsemble')

            model = HybridEnsemble(tabular_size=self.tabular_input_size)
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict):
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    logger.error(f"Failed to load state_dict into HybridEnsemble: {e}")
                    return False
            elif isinstance(state, torch.nn.Module):
                model = state
            else:
                logger.error(f"Unrecognized fusion model file contents: {type(state)}")
                return False

            model.to(self.device)
            model.eval()
            self.fusion_model = model
            self.is_loaded = True
            logger.info(f"Loaded fusion torch model from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load fusion model: {e}")
            return False

    def predict_tabular_array(self, features: List[float]) -> float:
        """Predict probability from a tabular feature list using the tabular model."""
        if self.tabular_model is None:
            raise RuntimeError("Tabular model not loaded")
        with torch.no_grad():
            tensor = torch.FloatTensor(features).view(1, -1).to(self.device)
            out = self.tabular_model(tensor)
            # If model outputs logits, apply sigmoid
            try:
                prob = torch.sigmoid(out).item()
            except Exception:
                prob = float(out.squeeze().cpu().numpy())
            return prob

    def predict_fusion(self, tabular_features: List[float], image_bytes: bytes = None) -> float:
        """Predict using fusion model; image_bytes is optional pixel-diff heatmap (png/jpg).
        If image_bytes is None, pass zeros for cnn inputs."""
        if self.fusion_model is None:
            raise RuntimeError("Fusion model not loaded")

        # Prepare tensors
        tab = torch.FloatTensor(tabular_features).view(1, -1).to(self.device)

        # Default cnn inputs
        cnn1d = torch.zeros(1, 5, 128, dtype=torch.float32).to(self.device)
        cnn2d = torch.zeros(1, 32, 24, 24, dtype=torch.float32).to(self.device)

        if image_bytes is not None:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert('L')
                arr = np.array(img, dtype=np.float32) / 255.0
                # Resize or pad to (24,24) and expand to phases dim if needed
                from skimage.transform import resize
                small = resize(arr, (24, 24), preserve_range=True)
                # Create fake phases dimension by repeating
                phases = np.stack([small for _ in range(32)], axis=0)  # (32,24,24)
                cnn2d = torch.FloatTensor(phases).unsqueeze(0).to(self.device)
            except Exception as e:
                logger.warning(f"Failed to process image for fusion model: {e}")

        with torch.no_grad():
            out, _ = self.fusion_model(tab, cnn1d, cnn2d)
            try:
                prob = torch.sigmoid(out).item()
            except Exception:
                prob = float(out.squeeze().cpu().numpy())
            return prob


# Global torch manager
torch_manager = TorchModelManager()

def initialize_torch_models() -> bool:
    # By default only load the tabular model (fusion models are large and
    # may have mismatched checkpoints). Set USE_FUSION=true in the env to
    # attempt to load the fusion model as well.
    ok_tab = torch_manager.load_tabular_model()
    use_fusion = str(os.getenv("USE_FUSION", "False")).lower() in ("1", "true", "yes")
    ok_fusion = False
    if use_fusion:
        ok_fusion = torch_manager.load_fusion_model()
    return ok_tab or ok_fusion

def get_torch_manager() -> TorchModelManager:
    return torch_manager


def request_to_feature_list(request_data: Any, target_length: int = None) -> List[float]:
    """Convert a dict or ExoplanetPredictionRequest to a numeric feature list.
    This is a best-effort mapping: it will take numeric fields in insertion order,
    encode `mission` to 0/1 (Kepler=0, TESS=1) and pad/truncate to target_length.
    """
    if target_length is None:
        target_length = EXO_TABULAR_FEATURES

    # If this is a Pydantic model, get dict
    if hasattr(request_data, 'dict'):
        data = request_data.dict()
    elif isinstance(request_data, dict):
        data = request_data
    else:
        raise ValueError('Unsupported request data type')

    features: List[float] = []

    # Handle mission encoding first if present
    if 'mission' in data:
        try:
            features.append(1.0 if str(data.get('mission', '')).upper() == 'TESS' else 0.0)
        except Exception:
            features.append(0.0)

    # Then iterate other numeric fields
    for k, v in data.items():
        if k == 'mission':
            continue
        if isinstance(v, (int, float)):
            features.append(float(v))
        else:
            # Attempt coercion
            try:
                features.append(float(v))
            except Exception:
                # fallback to default if available
                features.append(float(FEATURE_DEFAULTS.get(k, 0.0)))

    # Pad or truncate
    if len(features) < target_length:
        features += [0.0] * (target_length - len(features))
    elif len(features) > target_length:
        features = features[:target_length]

    return features
