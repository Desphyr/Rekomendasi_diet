"""
Predictor — Wrapper untuk memuat model RF dan melakukan inferensi
=================================================================
Menerima dict profil pengguna -> mengembalikan (diet_type, probabilities)

Catatan: Fitur Allergies TIDAK digunakan sesuai batasan sistem di laporan.
Mendukung semua fitur dataset:
  - Severity, Daily_Caloric_Intake, Glucose_mg_dL, Weekly_Exercise_Hours
  - Adherence_to_Diet_Plan, Dietary_Nutrient_Imbalance_Score
  - Dietary_Restrictions, Preferred_Cuisine
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/rf_diet_model.pkl")

ACTIVITY_ORDER       = ["Sedentary", "Light", "Moderate", "Active", "Very_Active"]
SEVERITY_ORDER       = ["Mild", "Moderate", "Severe"]
DISEASE_CLASSES      = ["Diabetes", "Hypertension", "Obesity", "None"]
RESTRICTION_CLASSES  = ["Low_Sugar", "Low_Sodium", "Low_Fat", "None"]
CUISINE_CLASSES      = ["Mexican", "Chinese", "Italian", "Indian", "Mediterranean", "None"]

NUMERICAL_FEATURES = [
    "Age", "Weight_kg", "Height_cm", "BMI",
    "Glucose_mg_dL",
    "Blood_Pressure_mmHg",
    "Cholesterol_mg_dL",
    "Daily_Caloric_Intake",
    "Weekly_Exercise_Hours",
    "Adherence_to_Diet_Plan",
    "Dietary_Nutrient_Imbalance_Score",
]


class DietPredictor:
    """Loads a trained Random Forest pipeline and performs diet type prediction."""

    def __init__(self, model_path: str = str(MODEL_PATH)):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di '{model_path}'. "
                "Jalankan 'python src/train.py' terlebih dahulu."
            )
        artifacts              = joblib.load(model_path)
        self.pipeline          = artifacts["pipeline"]
        self.label_encoder     = artifacts["label_encoder"]
        self.feature_cols      = artifacts["feature_cols"]
        # Ambil daftar kelas dari artefak (fallback ke konstanta default)
        self.disease_classes   = artifacts.get("disease_classes",    DISEASE_CLASSES)
        self.restriction_classes = artifacts.get("restriction_classes", RESTRICTION_CLASSES)
        self.cuisine_classes   = artifacts.get("cuisine_classes",    CUISINE_CLASSES)

    def predict(self, user_profile: dict) -> tuple[str, dict[str, float]]:
        """
        Memprediksi tipe diet dari profil pengguna.

        Returns
        -------
        diet_type     : str  — 'Balanced' | 'Low_Carb' | 'Low_Sodium'
        probabilities : dict — {diet_type: probability}
        """
        X = self._build_feature_vector(user_profile)
        pred_idx   = self.pipeline.predict(X)[0]
        pred_proba = self.pipeline.predict_proba(X)[0]

        diet_type  = self.label_encoder.inverse_transform([pred_idx])[0]
        proba_dict = {
            cls: float(p)
            for cls, p in zip(self.label_encoder.classes_, pred_proba)
        }
        return diet_type, proba_dict

    def _build_feature_vector(self, profile: dict) -> pd.DataFrame:
        """Mengonversi dict profil pengguna menjadi DataFrame fitur yang siap diprediksi."""
        row = {}

        # --- Fitur numerik ---------------------------------------------------
        for col in NUMERICAL_FEATURES:
            row[col] = profile.get(col, 0)

        # Sinkronisasi nama kolom lama ↔ baru (backward-compat)
        if row.get("Glucose_mg_dL", 0) == 0 and profile.get("Blood_Sugar_mgdL"):
            row["Glucose_mg_dL"] = profile["Blood_Sugar_mgdL"]
        if row.get("Blood_Pressure_mmHg", 0) == 0 and profile.get("Blood_Pressure_Systolic"):
            row["Blood_Pressure_mmHg"] = profile["Blood_Pressure_Systolic"]
        if row.get("Cholesterol_mg_dL", 0) == 0 and profile.get("Cholesterol_mgdL"):
            row["Cholesterol_mg_dL"] = profile["Cholesterol_mgdL"]

        # --- Gender → biner --------------------------------------------------
        row["Gender"] = 1 if profile.get("Gender", "Male") == "Male" else 0

        # --- Activity_Level (ordinal) ----------------------------------------
        row["Activity_Level"] = profile.get("Activity_Level", "Moderate")

        # --- Severity (ordinal) ---------------------------------------------
        row["Severity"] = profile.get("Severity", "Mild")

        # --- Disease_Type → One-Hot ------------------------------------------
        disease = profile.get("Disease_Type", "None")
        for d in self.disease_classes:
            row[f"Disease_{d}"] = 1 if disease == d else 0

        # --- Dietary_Restrictions → One-Hot ----------------------------------
        restriction = profile.get("Dietary_Restrictions", "None")
        for r in self.restriction_classes:
            row[f"Restrict_{r}"] = 1 if restriction == r else 0

        # --- Preferred_Cuisine → One-Hot -------------------------------------
        cuisine = profile.get("Preferred_Cuisine", "None")
        for c in self.cuisine_classes:
            row[f"Cuisine_{c}"] = 1 if cuisine == c else 0

        df = pd.DataFrame([row])

        # Pastikan kolom sesuai urutan training
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_cols]

        return df
