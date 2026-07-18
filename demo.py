import pandas as pd
from pathlib import Path

from src.predictor   import DietPredictor
from src.recommender import DietRecommender, DIET_RULES

DEMO_PROFILES = [
    {
        "name": "Kasus 1 — Pria Diabetes, BMI Tinggi",
        "profile": {
            "Age"                              : 52,
            "Gender"                           : "Male",
            "Weight_kg"                        : 85,
            "Height_cm"                        : 170,
            "BMI"                              : None,  # dihitung otomatis
            "Activity_Level"                   : "Light",
            "Weekly_Exercise_Hours"            : 2.0,
            "Disease_Type"                     : "Diabetes",
            "Severity"                         : "Moderate",
            "Blood_Sugar_mgdL"                 : 185,
            "Blood_Pressure_Systolic"          : 128,
            "Cholesterol_mgdL"                 : 210,
            "Daily_Caloric_Intake"             : 2400.0,
            "Adherence_to_Diet_Plan"           : 55.0,
            "Dietary_Nutrient_Imbalance_Score" : 4.5,
            "Dietary_Restrictions"             : "Low_Sugar",
            "Preferred_Cuisine"                : "None",
            "top_n_menus"                      : 3,
        }
    },
    {
        "name": "Kasus 2 — Wanita Hipertensi, Aktif",
        "profile": {
            "Age"                              : 45,
            "Gender"                           : "Female",
            "Weight_kg"                        : 65,
            "Height_cm"                        : 160,
            "BMI"                              : None,
            "Activity_Level"                   : "Moderate",
            "Weekly_Exercise_Hours"            : 5.0,
            "Disease_Type"                     : "Hypertension",
            "Severity"                         : "Mild",
            "Blood_Sugar_mgdL"                 : 95,
            "Blood_Pressure_Systolic"          : 145,
            "Cholesterol_mgdL"                 : 190,
            "Daily_Caloric_Intake"             : 1900.0,
            "Adherence_to_Diet_Plan"           : 78.0,
            "Dietary_Nutrient_Imbalance_Score" : 2.0,
            "Dietary_Restrictions"             : "Low_Sodium",
            "Preferred_Cuisine"                : "Mediterranean",
            "top_n_menus"                      : 3,
        }
    },
    {
        "name": "Kasus 3 — Pria Sehat, Aktif Berolahraga",
        "profile": {
            "Age"                              : 28,
            "Gender"                           : "Male",
            "Weight_kg"                        : 72,
            "Height_cm"                        : 175,
            "BMI"                              : None,
            "Activity_Level"                   : "Active",
            "Weekly_Exercise_Hours"            : 8.0,
            "Disease_Type"                     : "None",
            "Severity"                         : "Mild",
            "Blood_Sugar_mgdL"                 : 88,
            "Blood_Pressure_Systolic"          : 115,
            "Cholesterol_mgdL"                 : 165,
            "Daily_Caloric_Intake"             : 2700.0,
            "Adherence_to_Diet_Plan"           : 90.0,
            "Dietary_Nutrient_Imbalance_Score" : 1.5,
            "Dietary_Restrictions"             : "None",
            "Preferred_Cuisine"                : "None",
            "top_n_menus"                      : 3,
        }
    },
]


def compute_bmi(profile: dict) -> dict:
    p = profile.copy()
    if p.get("BMI") is None:
        h_m   = p["Height_cm"] / 100
        p["BMI"] = round(p["Weight_kg"] / (h_m ** 2), 2)

    # Backward compat
    if "Blood_Sugar_mgdL" in p and "Glucose_mg_dL" not in p:
        p["Glucose_mg_dL"] = p["Blood_Sugar_mgdL"]
    if "Blood_Pressure_Systolic" in p and "Blood_Pressure_mmHg" not in p:
        p["Blood_Pressure_mmHg"] = p["Blood_Pressure_Systolic"]
    if "Cholesterol_mgdL" in p and "Cholesterol_mg_dL" not in p:
        p["Cholesterol_mg_dL"] = p["Cholesterol_mgdL"]
    return p

def print_result(case_name: str, profile: dict, result, target_kcal: float):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {case_name}")
    print(sep)

    bmi = profile.get("BMI", 0)
    print(f"  Profil  : {profile['Gender']}, {profile['Age']} tahun | "
          f"BMI {bmi:.1f} | {profile['Disease_Type']}")
    print(f"  Olahraga: {profile['Weekly_Exercise_Hours']} jam/minggu | "
          f"Kepatuhan diet: {profile['Adherence_to_Diet_Plan']}%")
    print(f"\n  >> Tipe Diet   : {result.diet_type}")
    print(f"  >> Target Kalori: {target_kcal:.0f} kcal/hari")
    print(f"  >> Compliance  : {result.compliance_status}")

    prob = result.diet_probability.get(result.diet_type, 0)
    print(f"  >> Probabilitas: {prob*100:.1f}%")

    print(f"\n  Penjelasan Global:")
    for exp in result.global_explanation:
        print(f"    • {exp}")

    for i, menu in enumerate(result.recommended_menus, 1):
        print(f"\n  --- Pilihan Menu #{i} ---")
        for item in menu["menu_items"]:
            print(f"    🍽  {item}")
        n = menu["nutrition_summary"]
        print(f"    Nutrisi: {n['total_calories_kcal']} kcal | "
              f"Karbo {n['total_carbs_g']}g | Protein {n['total_protein_g']}g | "
              f"Lemak {n['total_fat_g']}g | Sodium {n['total_sodium_mg']}mg")
        print(f"    Skor   : {menu['score'] * 100:.1f}%")
        print(f"    Alasan :")
        for exp in menu["explanations"]:
            print(f"      - {exp}")

    print(sep)

def main():
    print("\n" + "=" * 65)
    print("  DSS DIET — Demo CLI")
    print("  Sistem Pendukung Keputusan Rekomendasi Menu Diet Harian")
    print("=" * 65)

    # Load model & database makanan
    try:
        predictor = DietPredictor()
        food_df   = pd.read_csv("data/food_nutrition.csv")
        print("\n[OK] Model Random Forest dan database makanan berhasil dimuat.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Pastikan Anda sudah menjalankan: python src/train.py")
        return

    # Jalankan setiap profil demo
    for case in DEMO_PROFILES:
        profile  = compute_bmi(case["profile"])
        top_n    = profile.pop("top_n_menus", 3)

        # Prediksi tipe diet
        diet_type, probabilities = predictor.predict(profile)

        # Buat rekomendasi menu
        recommender = DietRecommender(
            food_df        = food_df,
            top_n_menus    = top_n,
            target_calories= None,
        )
        result     = recommender.recommend(diet_type, probabilities, profile)
        target_kcal = recommender._estimate_calories(profile)

        print_result(case["name"], profile, result, target_kcal)

    print("\n[OK] Demo selesai.\n")


if __name__ == "__main__":
    main()
