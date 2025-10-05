## 🪐 Exoplanet Classification Model API Integration

This document outlines how to integrate the trained Random Forest classification model (`rf_combined_model.joblib`) into a backend API (like FastAPI or Flask) to process new data, handle missing inputs, and return classification probabilities.

-----

## I. Model Artifacts Reference

The `trained_model/` directory contains the five critical files needed to run the prediction pipeline. These files **must be loaded once at API startup** and then used for every incoming prediction request.

| Filename | Type | Purpose |
| :--- | :--- | :--- |
| **`rf_combined_model.joblib`** | Trained Model | The final, optimized **Random Forest Classifier**. Used to generate the prediction (class) and confidence (probability). |
| **`scaler_combined.joblib`** | Preprocessor | The fitted **`StandardScaler`**. Used to normalize (scale) all input features before prediction. |
| **`label_encoder_combined.joblib`** | Encoder | The fitted **`LabelEncoder`**. Used to convert the model's numeric output (0 or 1) back into human-readable labels: **CONFIRMED** or **FALSE POSITIVE**. |
| **`imputer_medians_combined.joblib`** | Preprocessor | A dictionary of **median values** calculated from the training data. Used to fill any missing inputs provided by the user (e.g., if `planet_radius_re` is missing, it is replaced by the median radius). |
| **`feature_columns_combined.txt`** | Reference | A list of all $\mathbf{13}$ input features (including 2 engineered features) in the **EXACT ORDER** the model was trained on. This is essential for preventing the `ValueError: Feature names mismatch` error. |

-----

## II. API Integration Logic

The prediction endpoint (e.g., a `/classify_csv` endpoint) must perform **all four steps** below, in sequence, to correctly prepare the user's uploaded data.

### A. Expected Input Format (CSV Upload)

The user's uploaded CSV file **must** contain columns corresponding to the following **11 base input features**. Any omitted column or missing value should be represented as a **NaN** or empty cell.

| Required Feature (User Input Header) | Internal Feature Name |
| :--- | :--- |
| `orbital_period_days` | `ORB_PERIOD` |
| `transit_duration_hours` | `transit_duration_hours` |
| `transit_depth_ppm` | `TRANSIT_DEPTH` |
| `planet_radius_re` | `PLANET_RADIUS` |
| `equilibrium_temp_k` | `equilibrium_temp_k` |
| `insolation_flux_earth` | `INSOL_FLUX` |
| `stellar_teff_k` | `stellar_teff_k` |
| `stellar_radius_re` | `STELLAR_RADIUS` |
| `apparent_mag` | `apparent_mag` |
| `ra` | `ra` |
| `dec` | `dec` |

### B. The 4-Step Preprocessing Pipeline

The Python function handling the prediction request must follow this flow:

| Step | Action | Required Artifact(s) |
| :--- | :--- | :--- |
| **1. Data Alignment** | **Align columns:** Create a DataFrame using the user's data, ensuring it contains **all** columns listed in `feature_columns_combined.txt` (including the temporary names like `PLANET_RADIUS`). Missing columns or missing cell values are initialized to `NaN`. | `feature_columns_combined.txt` |
| **2. Imputation & Engineering** | **Fill NaNs:** Replace all `NaN` values with the median values from `imputer_medians_combined.joblib`. **Calculate Engineered Features:** Create `R_PLANET_R_STAR_RATIO` and `DEPTH_PER_RADIUS` using the now-imputed feature values. **Apply Log Transforms** (e.g., `np.log1p`). | `imputer_medians_combined.joblib`|
| **3. Scaling & Order** | **Scale:** Apply `scaler_combined.joblib.transform()` to the complete, engineered feature array. **Order:** Ensure the final 2D numpy array is in the exact feature order defined in the `.txt` file. | `scaler_combined.joblib`, `feature_columns_combined.txt` |
| **4. Prediction** | Call **`rf_combined_model.predict_proba(X_scaled)`** to retrieve the confidence scores for each target. | `rf_combined_model.joblib`, `label_encoder_combined.joblib`|

### C. Recommended API Output

The API should return the original data merged with the calculated results, formatted as JSON:

```json
[
  {
    "id": 12345,
    "orbital_period_days": 10.5,
    "...": "...",
    "Predicted_Disposition": "CONFIRMED",
    "Confidence_Score": 0.965, // The highest probability value
    "Prob_CONFIRMED": 0.965,
    "Prob_FALSE_POSITIVE": 0.035
  },
  // ... more predictions
]
```