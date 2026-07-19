"""
Training Pipeline — Random Forest Classifier
=============================================
Tugas   : Memprediksi tipe diet optimal berdasarkan profil pengguna
Target  : Diet_Recommendation  (Balanced | Low_Carb | Low_Sodium)
Output  : models/rf_diet_model.pkl

Dataset : diet_recommendations.csv (20 kolom, 1000 pasien)
Fitur yang Digunakan:
  - Severity                     : Tingkat keparahan penyakit (Mild/Moderate/Severe)
  - Daily_Caloric_Intake         : Asupan kalori harian aktual
  - Glucose_mg_dL                : Kadar glukosa darah
  - Cholesterol_mg_dL            : Kadar kolesterol
  - Blood_Pressure_mmHg          : Tekanan darah sistolik
  - Weekly_Exercise_Hours        : Jam olahraga per minggu
  - Adherence_to_Diet_Plan       : Kepatuhan terhadap rencana diet (%)
  - Dietary_Nutrient_Imbalance_Score : Skor ketidakseimbangan nutrisi
  - Dietary_Restrictions         : Batasan diet (Low_Sugar/Low_Sodium/None)
  - Preferred_Cuisine            : Preferensi masakan

Catatan: Fitur Allergies TIDAK digunakan sesuai batasan sistem di laporan.
Ablation Study: Model dilatih dengan 2 varian (dengan dan tanpa Disease_Type)
  untuk menganalisis sensitivitas performa terhadap atribut tersebut.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

DATA_PATH  = "data/diet_recommendations.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fitur numerik
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
CUISINE_CLASSES      = ["Mexican", "Chinese", "Italian", "Indian", "Mediterranean", "None"]

TARGET = "Diet_Recommendation"

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
    df["Preferred_Cuisine"]     = df["Preferred_Cuisine"].fillna("None")
    df["Severity"]              = df["Severity"].fillna("Mild")

    # Hapus kolom Allergies (tidak digunakan sesuai batasan sistem)
    if "Allergies" in df.columns:
        df = df.drop(columns=["Allergies"])

    print(f"Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("\nDistribusi label:\n", df[TARGET].value_counts())
    print("\nKolom tersedia:\n", list(df.columns))
    return df

def encode_categoricals(df: pd.DataFrame, include_disease: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode semua fitur kategorikal:
      - Gender         → biner (0/1)
      - Disease_Type   → One-Hot (opsional, dipakai untuk ablation study)
      - Dietary_Restrictions → One-Hot
      - Preferred_Cuisine → One-Hot
    Kembalikan df_encoded dan daftar kolom dummy yang dibuat.
    """
    df = df.copy()

    # Gender → biner
    df["Gender"] = (df["Gender"] == "Male").astype(int)

    dummy_cols = []

    # Disease_Type → One-Hot (dikendalikan oleh flag include_disease)
    if include_disease and "Disease_Type" in df.columns:
        df = pd.get_dummies(df, columns=["Disease_Type"], prefix="Disease", dtype=int)
        for dis in DISEASE_CLASSES:
            col = f"Disease_{dis}"
            if col not in df.columns:
                df[col] = 0
        dummy_cols += [f"Disease_{d}" for d in DISEASE_CLASSES]
    elif "Disease_Type" in df.columns:
        df = df.drop(columns=["Disease_Type"])

    # Dietary_Restrictions → One-Hot
    df = pd.get_dummies(df, columns=["Dietary_Restrictions"], prefix="Restrict", dtype=int)
    for r in RESTRICTION_CLASSES:
        col = f"Restrict_{r}"
        if col not in df.columns:
            df[col] = 0
    dummy_cols += [f"Restrict_{r}" for r in RESTRICTION_CLASSES]

    # Preferred_Cuisine → One-Hot
    df = pd.get_dummies(df, columns=["Preferred_Cuisine"], prefix="Cuisine", dtype=int)
    for c in CUISINE_CLASSES:
        col = f"Cuisine_{c}"
        if col not in df.columns:
            df[col] = 0
    dummy_cols += [f"Cuisine_{c}" for c in CUISINE_CLASSES]

    return df, dummy_cols

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
# 4. Training (satu varian)
# -----------------------------------------------------------------------------
def _run_training(df: pd.DataFrame, classifier=None, include_disease: bool = True, label: str = "") -> dict:
    """Jalankan training pipeline dan kembalikan artefak + metrik."""
    df_enc, dummy_cols = encode_categoricals(df, include_disease=include_disease)

    base_features = NUMERICAL_FEATURES + ["Activity_Level", "Severity", "Gender"]
    feature_cols = base_features + [c for c in dummy_cols if c in df_enc.columns]
    feature_cols = [c for c in feature_cols if c in df_enc.columns]

    X = df_enc[feature_cols]
    y = df_enc[TARGET]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    tag = f"[{label}] " if label else ""
    print(f"\n{tag}Jumlah fitur: {len(feature_cols)}")
    print(f"{tag}Kelas target:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    preprocessor = build_preprocessor()

    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=500,
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
        ("classifier", classifier)
    ])

    # Cross-validation
    print(f"\n{tag}-- Cross-Validation (5-Fold Stratified) ----------------------")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"{tag}CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Fit final model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{tag}-- Evaluasi Test Set ------------------------------------------")
    print(f"{tag}Accuracy : {acc:.4f}")
    print(f"\n{tag}Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"{tag}Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "pipeline"            : pipeline,
        "label_encoder"       : le,
        "feature_cols"        : feature_cols,
        "dummy_cols"          : dummy_cols,
        "disease_classes"     : DISEASE_CLASSES,
        "restriction_classes" : RESTRICTION_CLASSES,
        "cuisine_classes"     : CUISINE_CLASSES,
        "cv_mean"             : cv_scores.mean(),
        "test_accuracy"       : acc,
    }

def run_ablation_study(df: pd.DataFrame):
    """
    Ablation study untuk menganalisis sensitivitas performa model
    terhadap ada/tidaknya atribut Disease_Type.
    Sesuai dengan rencana evaluasi di laporan penelitian.
    """
    print("\n" + "=" * 60)
    print("  ABLATION STUDY: dengan vs tanpa Disease_Type")
    print("=" * 60)

    print("\n>>> Varian A: Model DENGAN Disease_Type")
    result_with    = _run_training(df, include_disease=True,  label="WITH Disease_Type")

    print("\n>>> Varian B: Model TANPA Disease_Type")
    result_without = _run_training(df, include_disease=False, label="WITHOUT Disease_Type")

    print("\n" + "=" * 60)
    print("  HASIL PERBANDINGAN ABLATION STUDY")
    print("=" * 60)
    print(f"  {'Varian':<35} {'CV Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Dengan Disease_Type':<35} {result_with['cv_mean']:>9.4f}  {result_with['test_accuracy']:>9.4f}")
    print(f"  {'Tanpa Disease_Type':<35} {result_without['cv_mean']:>9.4f}  {result_without['test_accuracy']:>9.4f}")

    diff = result_with["test_accuracy"] - result_without["test_accuracy"]
    print(f"\n  Selisih akurasi (dengan - tanpa): {diff:+.4f}")
    if abs(diff) < 0.02:
        print("  -> Dampak Disease_Type relatif KECIL (< 2%): model robust tanpa atribut ini.")
    else:
        print(f"  -> Disease_Type berkontribusi signifikan ({abs(diff)*100:.1f}%) terhadap akurasi.")
    print("=" * 60)

    return result_with, result_without

def run_algorithm_comparison(df: pd.DataFrame):
    """Bandingkan performa Random Forest dengan Decision Tree & Logistic Regression"""
    print("\n" + "=" * 60)
    print("  PERBANDINGAN ALGORITMA KLASIFIKASI")
    print("=" * 60)

    print("\n>>> 1. Random Forest (Default Model)")
    rf_clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
    res_rf = _run_training(df, classifier=rf_clf, label="Random Forest")

    print("\n>>> 2. Decision Tree")
    dt_clf = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    res_dt = _run_training(df, classifier=dt_clf, label="Decision Tree")

    print("\n>>> 3. Logistic Regression")
    lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    res_lr = _run_training(df, classifier=lr_clf, label="Logistic Regression")

    print("\n" + "=" * 60)
    print("  HASIL PERBANDINGAN ALGORITMA")
    print("=" * 60)
    print(f"  {'Algoritma':<25} {'CV Acc':>10} {'Test Acc':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Random Forest':<25} {res_rf['cv_mean']:>9.4f}  {res_rf['test_accuracy']:>9.4f}")
    print(f"  {'Decision Tree':<25} {res_dt['cv_mean']:>9.4f}  {res_dt['test_accuracy']:>9.4f}")
    print(f"  {'Logistic Regression':<25} {res_lr['cv_mean']:>9.4f}  {res_lr['test_accuracy']:>9.4f}")
    print("=" * 60)

def train(df: pd.DataFrame):
    """Training penuh: jalankan ablation study lalu simpan model utama (dengan Disease_Type)."""

    # Perbandingan Algoritma
    run_algorithm_comparison(df)

    # Ablation study
    result_with, result_without = run_ablation_study(df)

    # Simpan model utama (varian lengkap dengan Disease_Type)
    artifacts = {k: v for k, v in result_with.items()
                 if k not in ("cv_mean", "test_accuracy")}
    joblib.dump(artifacts, f"{MODEL_DIR}/rf_diet_model.pkl")
    print(f"\n[OK] Model utama (dengan Disease_Type) disimpan -> {MODEL_DIR}/rf_diet_model.pkl")

    # Feature Importance (dari model utama)
    rf_fitted = result_with["pipeline"].named_steps["classifier"]
    feature_cols = result_with["feature_cols"]
    remainder_names = [c for c in feature_cols
                       if c not in NUMERICAL_FEATURES + ["Activity_Level", "Severity"]]
    transformed_feature_names = (
        NUMERICAL_FEATURES + ["Activity_Level", "Severity"] + remainder_names
    )

    importances = rf_fitted.feature_importances_
    n_features  = min(len(transformed_feature_names), len(importances))
    fi_df = pd.DataFrame({
        "Feature"   : transformed_feature_names[:n_features],
        "Importance": importances[:n_features],
    }).sort_values("Importance", ascending=False)

    print("\n-- Top 15 Feature Importances -----------------------------------")
    print(fi_df.head(15).to_string(index=False))

    return result_with

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train(df)
