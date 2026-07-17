"""
Training Pipeline — Random Forest Classifier
=============================================
Tugas   : Memprediksi tipe diet optimal berdasarkan profil pengguna
Target  : Diet_Recommendation  (Balanced | Low_Carb | Low_Sodium)
Output  : models/rf_diet_model.pkl  +  models/preprocessor.pkl

Dataset : diet_recommendations_dataset.csv (20 kolom, 1001 pasien)
Fitur Baru yang Dimanfaatkan:
  - Severity                     : Tingkat keparahan penyakit (Mild/Moderate/Severe)
  - Daily_Caloric_Intake         : Asupan kalori harian aktual
  - Glucose_mg/dL                : Kadar glukosa darah
  - Cholesterol_mg/dL            : Kadar kolesterol
  - Blood_Pressure_mmHg          : Tekanan darah sistolik
  - Weekly_Exercise_Hours        : Jam olahraga per minggu
  - Adherence_to_Diet_Plan       : Kepatuhan terhadap rencana diet (%)
  - Dietary_Nutrient_Imbalance_Score : Skor ketidakseimbangan nutrisi
  - Dietary_Restrictions         : Batasan diet (Low_Sugar/Low_Sodium/None)
  - Allergies                    : Alergi makanan
  - Preferred_Cuisine            : Preferensi masakan
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# -----------------------------------------------------------------------------
# Konstanta
# -----------------------------------------------------------------------------
DATA_PATH  = "data/diet_recommendations.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fitur numerik — termasuk semua kolom baru yang bermakna secara klinis
NUMERICAL_FEATURES = [
    "Age", "Weight_kg", "Height_cm", "BMI",
    "Glucose_mg_dL",           # gula darah (rename dari Glucose_mg/dL)
    "Blood_Pressure_mmHg",     # tekanan darah sistolik
    "Cholesterol_mg_dL",       # kolesterol (rename dari Cholesterol_mg/dL)
    "Daily_Caloric_Intake",    # asupan kalori harian aktual
    "Weekly_Exercise_Hours",   # jam olahraga per minggu
    "Adherence_to_Diet_Plan",  # kepatuhan diet (%)
    "Dietary_Nutrient_Imbalance_Score",  # skor ketidakseimbangan nutrisi
]

# Fitur kategorikal ordinal
ACTIVITY_ORDER  = ["Sedentary", "Light", "Moderate", "Active", "Very_Active"]
SEVERITY_ORDER  = ["Mild", "Moderate", "Severe"]

# Fitur kategorikal nominal (dummies)
DISEASE_CLASSES      = ["Diabetes", "Hypertension", "Obesity", "None"]
RESTRICTION_CLASSES  = ["Low_Sugar", "Low_Sodium", "Low_Fat", "None"]
ALLERGY_CLASSES      = ["Peanuts", "Gluten", "Lactose", "Shellfish", "None"]
CUISINE_CLASSES      = ["Mexican", "Chinese", "Italian", "Indian", "Mediterranean", "None"]

TARGET = "Diet_Recommendation"


# -----------------------------------------------------------------------------
# 1. Load & bersihkan data
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Rename kolom dengan karakter khusus (/) agar kompatibel
    df = df.rename(columns={
        "Glucose_mg/dL"     : "Glucose_mg_dL",
        "Cholesterol_mg/dL" : "Cholesterol_mg_dL",
        "Physical_Activity_Level": "Activity_Level",
    })

    # Hapus kolom Patient_ID (bukan fitur prediktif)
    if "Patient_ID" in df.columns:
        df = df.drop(columns=["Patient_ID"])

    # Isi nilai hilang
    df["Dietary_Restrictions"] = df["Dietary_Restrictions"].fillna("None")
    df["Allergies"]             = df["Allergies"].fillna("None")
    df["Preferred_Cuisine"]     = df["Preferred_Cuisine"].fillna("None")
    df["Severity"]              = df["Severity"].fillna("Mild")

    print(f"Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("\nDistribusi label:\n", df[TARGET].value_counts())
    print("\nKolom tersedia:\n", list(df.columns))
    return df


# -----------------------------------------------------------------------------
# 2. Encode kategorikal sebelum pipeline
# -----------------------------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode semua fitur kategorikal:
      - Gender         → biner (0/1)
      - Disease_Type   → One-Hot
      - Dietary_Restrictions → One-Hot
      - Allergies      → One-Hot
      - Preferred_Cuisine → One-Hot (opsional, kontribusi kecil)
    Kembalikan df_encoded dan daftar kolom dummy yang dibuat.
    """
    df = df.copy()

    # Gender → biner
    df["Gender"] = (df["Gender"] == "Male").astype(int)

    # Disease_Type → One-Hot
    df = pd.get_dummies(df, columns=["Disease_Type"], prefix="Disease", dtype=int)
    for dis in DISEASE_CLASSES:
        col = f"Disease_{dis}"
        if col not in df.columns:
            df[col] = 0

    # Dietary_Restrictions → One-Hot
    df = pd.get_dummies(df, columns=["Dietary_Restrictions"], prefix="Restrict", dtype=int)
    for r in RESTRICTION_CLASSES:
        col = f"Restrict_{r}"
        if col not in df.columns:
            df[col] = 0

    # Allergies → One-Hot
    df = pd.get_dummies(df, columns=["Allergies"], prefix="Allergy", dtype=int)
    for a in ALLERGY_CLASSES:
        col = f"Allergy_{a}"
        if col not in df.columns:
            df[col] = 0

    # Preferred_Cuisine → One-Hot
    df = pd.get_dummies(df, columns=["Preferred_Cuisine"], prefix="Cuisine", dtype=int)
    for c in CUISINE_CLASSES:
        col = f"Cuisine_{c}"
        if col not in df.columns:
            df[col] = 0

    # Kumpulkan semua kolom dummy
    dummy_cols = (
        [f"Disease_{d}"  for d in DISEASE_CLASSES]
        + [f"Restrict_{r}" for r in RESTRICTION_CLASSES]
        + [f"Allergy_{a}"  for a in ALLERGY_CLASSES]
        + [f"Cuisine_{c}"  for c in CUISINE_CLASSES]
    )

    return df, dummy_cols


# -----------------------------------------------------------------------------
# 3. Bangun ColumnTransformer
# -----------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """
    - Numerik         : StandardScaler
    - Activity_Level  : OrdinalEncoder (Sedentary < Light < Moderate < Active < Very_Active)
    - Severity        : OrdinalEncoder (Mild < Moderate < Severe)
    - Semua dummy     : passthrough (sudah 0/1)
    """
    numerical_transformer = StandardScaler()

    activity_encoder = OrdinalEncoder(
        categories=[ACTIVITY_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    severity_encoder = OrdinalEncoder(
        categories=[SEVERITY_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",      numerical_transformer, NUMERICAL_FEATURES),
            ("activity", activity_encoder,      ["Activity_Level"]),
            ("severity", severity_encoder,      ["Severity"]),
        ],
        remainder="passthrough"   # Gender + semua dummy masuk tanpa transformasi
    )
    return preprocessor


# -----------------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------------
def train(df: pd.DataFrame):
    # Encode kategorik
    df_enc, dummy_cols = encode_categoricals(df)

    # Susun daftar fitur final
    feature_cols = (
        NUMERICAL_FEATURES
        + ["Activity_Level", "Severity", "Gender"]
        + [c for c in dummy_cols if c in df_enc.columns]
    )

    # Filter kolom yang benar-benar ada di dataframe
    feature_cols = [c for c in feature_cols if c in df_enc.columns]

    X = df_enc[feature_cols]
    y = df_enc[TARGET]

    # Encode label target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nJumlah fitur: {len(feature_cols)}")
    print("Kelas target:", list(le.classes_))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Bangun preprocessor
    preprocessor = build_preprocessor()

    # Pipeline penuh: preprocessor + classifier
    rf = RandomForestClassifier(
        n_estimators=500,          # lebih banyak pohon untuk dataset yang lebih kaya
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf)
    ])

    # -- Cross-validation ------------------------------------------------------
    print("\n-- Cross-Validation (5-Fold Stratified) -------------------------")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -- Fit final model -------------------------------------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # -- Evaluasi --------------------------------------------------------------
    print("\n-- Evaluasi Test Set --------------------------------------------")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -- Feature Importance ----------------------------------------------------
    rf_fitted = pipeline.named_steps["classifier"]
    try:
        # Coba ambil nama fitur dari ColumnTransformer
        num_names      = NUMERICAL_FEATURES
        activity_names = ["Activity_Level"]
        severity_names = ["Severity"]
        remainder_names = [c for c in feature_cols
                           if c not in NUMERICAL_FEATURES + ["Activity_Level", "Severity"]]
        transformed_feature_names = num_names + activity_names + severity_names + remainder_names
    except Exception:
        transformed_feature_names = feature_cols

    importances = rf_fitted.feature_importances_
    n_features  = min(len(transformed_feature_names), len(importances))
    fi_df = pd.DataFrame({
        "Feature"   : transformed_feature_names[:n_features],
        "Importance": importances[:n_features],
    }).sort_values("Importance", ascending=False)

    print("\n-- Top 15 Feature Importances -----------------------------------")
    print(fi_df.head(15).to_string(index=False))

    # -- Simpan artefak --------------------------------------------------------
    artifacts = {
        "pipeline"     : pipeline,
        "label_encoder": le,
        "feature_cols" : feature_cols,
        "dummy_cols"   : dummy_cols,
        "disease_classes"     : DISEASE_CLASSES,
        "restriction_classes" : RESTRICTION_CLASSES,
        "allergy_classes"     : ALLERGY_CLASSES,
        "cuisine_classes"     : CUISINE_CLASSES,
    }
    joblib.dump(artifacts, f"{MODEL_DIR}/rf_diet_model.pkl")
    print(f"\n[OK] Model disimpan -> {MODEL_DIR}/rf_diet_model.pkl")
    return artifacts


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train(df)
