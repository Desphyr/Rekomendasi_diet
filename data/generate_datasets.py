"""
Generate Datasets — Pembuat Dataset Sintetis untuk DSS Diet
============================================================
Script ini menghasilkan dua dataset yang digunakan oleh sistem:

  1. diet_recommendations.csv
     Dataset profil pasien untuk melatih model Random Forest.
     Berisi 1000 baris dengan 20 kolom fitur klinis dan gaya hidup.

  2. food_nutrition.csv
     Database makanan per porsi lengkap dengan informasi nutrisi.
     Berisi menu sarapan, makan siang, makan malam, dan camilan.

Cara menjalankan:
    python data/generate_datasets.py

Output akan disimpan di folder data/.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Seed untuk reprodusibilitas
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent
N_PATIENTS = 1000


# =============================================================================
# 1. Generate diet_recommendations.csv
# =============================================================================
def generate_diet_recommendations(n: int = N_PATIENTS) -> pd.DataFrame:
    """
    Buat dataset profil pengguna sintetis dengan label Diet_Recommendation.
    Label ditentukan berdasarkan aturan klinis yang masuk akal:
      - Diabetes / Glucose >= 126 → Low_Carb
      - Hypertension / BP >= 140  → Low_Sodium
      - Lainnya                   → Balanced
    """
    records = []

    for i in range(n):
        age    = np.random.randint(18, 75)
        gender = np.random.choice(["Male", "Female"])
        weight = round(np.random.uniform(45, 120), 1)
        height = round(np.random.uniform(150, 190), 1)
        bmi    = round(weight / (height / 100) ** 2, 2)

        activity = np.random.choice(
            ["Sedentary", "Light", "Moderate", "Active", "Very_Active"],
            p=[0.20, 0.25, 0.30, 0.15, 0.10]
        )

        # Kondisi medis
        disease = np.random.choice(
            ["Diabetes", "Hypertension", "Obesity", "None"],
            p=[0.20, 0.25, 0.15, 0.40]
        )
        severity = "Mild"
        if disease != "None":
            severity = np.random.choice(["Mild", "Moderate", "Severe"], p=[0.50, 0.35, 0.15])

        # Biomarker klinis (nilai disesuaikan dengan kondisi medis)
        if disease == "Diabetes":
            glucose = round(np.random.uniform(126, 350), 1)
            bp      = np.random.randint(100, 155)
            chol    = round(np.random.uniform(170, 280), 1)
        elif disease == "Hypertension":
            glucose = round(np.random.uniform(80, 130), 1)
            bp      = np.random.randint(140, 200)
            chol    = round(np.random.uniform(180, 300), 1)
        elif disease == "Obesity":
            glucose = round(np.random.uniform(90, 170), 1)
            bp      = np.random.randint(110, 160)
            chol    = round(np.random.uniform(190, 320), 1)
        else:
            glucose = round(np.random.uniform(70, 120), 1)
            bp      = np.random.randint(90, 135)
            chol    = round(np.random.uniform(140, 220), 1)

        # Gaya hidup
        exercise_hours = round(np.random.uniform(0, 14), 1)
        caloric_intake = round(np.random.uniform(1200, 3500), 0)
        adherence      = round(np.random.uniform(20, 100), 1)
        imbalance      = round(np.random.uniform(0, 10), 1)

        # Preferensi dan batasan
        restriction = np.random.choice(
            ["None", "Low_Sugar", "Low_Sodium", "Low_Fat"],
            p=[0.55, 0.15, 0.20, 0.10]
        )
        cuisine = np.random.choice(
            ["None", "Mexican", "Chinese", "Italian", "Indian", "Mediterranean"],
            p=[0.40, 0.10, 0.15, 0.10, 0.10, 0.15]
        )

        # Label rekomendasi diet
        if disease == "Diabetes" or glucose >= 126:
            label = "Low_Carb"
        elif disease == "Hypertension" or bp >= 140:
            label = "Low_Sodium"
        else:
            label = "Balanced"

        records.append({
            "Patient_ID"                      : f"P{i+1:04d}",
            "Age"                             : age,
            "Gender"                          : gender,
            "Weight_kg"                       : weight,
            "Height_cm"                       : height,
            "BMI"                             : bmi,
            "Physical_Activity_Level"         : activity,
            "Weekly_Exercise_Hours"           : exercise_hours,
            "Disease_Type"                    : disease,
            "Severity"                        : severity,
            "Glucose_mg/dL"                   : glucose,
            "Blood_Pressure_mmHg"             : bp,
            "Cholesterol_mg/dL"               : chol,
            "Daily_Caloric_Intake"            : caloric_intake,
            "Adherence_to_Diet_Plan"          : adherence,
            "Dietary_Nutrient_Imbalance_Score": imbalance,
            "Dietary_Restrictions"            : restriction,
            "Preferred_Cuisine"               : cuisine,
            "Diet_Recommendation"             : label,
        })

    df = pd.DataFrame(records)
    print(f"[OK] diet_recommendations.csv: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"     Distribusi label:\n{df['Diet_Recommendation'].value_counts().to_string()}")
    return df


# =============================================================================
# 2. Generate food_nutrition.csv
# =============================================================================
FOOD_DATABASE = [
    # Breakfast
    ("Oatmeal with Berries",        "Breakfast", 320, 55, 12, 8,  320, 6.0, 1.5),
    ("Scrambled Eggs",              "Breakfast", 280, 5,  20, 20, 380, 0.5, 6.0),
    ("Whole Wheat Toast with Avocado","Breakfast",350, 38, 10, 18, 290, 8.0, 2.5),
    ("Greek Yogurt Parfait",        "Breakfast", 290, 40, 18, 6,  210, 2.0, 1.5),
    ("Banana Smoothie",             "Breakfast", 260, 50, 8,  4,  150, 4.0, 1.0),
    ("Veggie Omelette",             "Breakfast", 310, 8,  22, 22, 410, 2.5, 6.5),
    ("Quinoa Porridge",             "Breakfast", 340, 52, 14, 7,  180, 5.0, 1.0),
    ("Whole Grain Cereal with Milk","Breakfast", 300, 55, 10, 5,  280, 4.0, 1.5),
    ("Poached Eggs on Toast",       "Breakfast", 320, 30, 18, 14, 450, 2.0, 4.0),
    ("Fresh Fruit Bowl",            "Breakfast", 180, 42, 3,  1,  30,  5.5, 0.2),
    ("Cottage Cheese with Fruit",   "Breakfast", 240, 28, 22, 4,  350, 2.0, 1.5),
    ("Almond Butter Toast",         "Breakfast", 360, 32, 12, 20, 180, 4.5, 2.5),

    # Lunch
    ("Grilled Chicken Salad",       "Lunch", 420, 20, 40, 18, 480, 6.0,  3.0),
    ("Brown Rice with Steamed Veggies","Lunch",380, 72, 10, 4,  210, 7.0, 0.5),
    ("Lentil Soup",                 "Lunch", 350, 58, 20, 4,  520, 12.0, 0.5),
    ("Turkey Sandwich Whole Grain", "Lunch", 430, 48, 30, 12, 680, 5.0,  3.0),
    ("Tuna Salad",                  "Lunch", 380, 15, 35, 18, 520, 3.5,  2.5),
    ("Veggie Stir Fry with Tofu",   "Lunch", 360, 40, 20, 14, 490, 6.5,  2.0),
    ("Chicken Caesar Salad",        "Lunch", 440, 18, 38, 22, 710, 4.0,  5.0),
    ("Chickpea Curry with Rice",    "Lunch", 480, 78, 18, 10, 540, 9.0,  1.5),
    ("Salmon with Quinoa",          "Lunch", 520, 42, 40, 18, 380, 5.0,  3.0),
    ("Black Bean Bowl",             "Lunch", 410, 62, 20, 8,  420, 14.0, 1.0),
    ("Mediterranean Wrap",          "Lunch", 460, 55, 22, 16, 630, 6.0,  3.5),
    ("Vegetable Soup with Bread",   "Lunch", 320, 52, 12, 6,  680, 8.0,  1.0),
    ("Grilled Fish Tacos",          "Lunch", 450, 48, 32, 14, 590, 5.0,  2.5),

    # Dinner
    ("Baked Salmon with Broccoli",  "Dinner", 480, 15, 40, 28, 450, 6.0, 5.0),
    ("Grilled Chicken with Sweet Potato","Dinner",500,55, 42, 12, 520, 7.0, 2.5),
    ("Beef Stir Fry with Vegetables","Dinner",520, 32, 38, 24, 680, 5.0, 8.0),
    ("Vegetarian Pasta Primavera",  "Dinner", 460, 72, 18, 10, 380, 7.5, 2.0),
    ("Baked Chicken with Quinoa",   "Dinner", 490, 45, 42, 12, 440, 5.5, 2.5),
    ("Shrimp with Brown Rice",      "Dinner", 450, 52, 32, 10, 620, 4.0, 1.5),
    ("Lamb Chops with Roasted Veg", "Dinner", 560, 28, 40, 30, 580, 6.0, 12.0),
    ("Tofu Coconut Curry",          "Dinner", 430, 48, 20, 18, 520, 6.5, 8.0),
    ("Baked Cod with Asparagus",    "Dinner", 380, 18, 38, 12, 490, 5.0, 2.0),
    ("Turkey Meatballs with Marinara","Dinner",510, 42, 40, 18, 720, 6.0, 5.0),
    ("Stuffed Bell Peppers",        "Dinner", 420, 45, 28, 14, 560, 8.0, 3.5),
    ("Grilled Tuna Steak",          "Dinner", 470, 10, 45, 26, 400, 2.0, 6.0),
    ("Chicken Tikka Masala",        "Dinner", 530, 38, 40, 22, 650, 4.0, 7.0),

    # Snack
    ("Apple with Peanut Butter",    "Snack", 200, 28, 6,  10, 95,  4.0, 2.0),
    ("Mixed Nuts",                  "Snack", 180, 8,  5,  16, 90,  2.5, 2.0),
    ("Hummus with Carrots",         "Snack", 150, 20, 6,  6,  280, 4.5, 1.0),
    ("Low Fat Greek Yogurt",        "Snack", 130, 12, 15, 2,  75,  0.5, 1.0),
    ("Celery with Almond Butter",   "Snack", 140, 10, 4,  10, 80,  3.0, 1.0),
    ("Edamame",                     "Snack", 120, 10, 11, 5,  15,  4.0, 0.5),
    ("Rice Cakes with Avocado",     "Snack", 160, 22, 3,  8,  110, 3.5, 1.0),
    ("Cottage Cheese",              "Snack", 110, 5,  14, 3,  380, 0.2, 1.5),
    ("Boiled Egg",                  "Snack", 78,  1,  6,  5,  62,  0.0, 1.6),
    ("Banana",                      "Snack", 105, 27, 1,  0,  1,   3.1, 0.1),
]

FOOD_COLUMNS = [
    "Food_Name", "Meal_Type", "Calories_kcal", "Carbs_g",
    "Protein_g", "Fat_g", "Sodium_mg", "Fiber_g", "Saturated_Fat_g"
]


def generate_food_nutrition() -> pd.DataFrame:
    """Buat database makanan dari daftar statis dengan variasi kecil."""
    rows = []
    for item in FOOD_DATABASE:
        # Tambahkan variasi kecil (±5%) untuk keragaman data
        cal  = round(item[2] * np.random.uniform(0.95, 1.05), 1)
        carb = round(item[3] * np.random.uniform(0.95, 1.05), 1)
        prot = round(item[4] * np.random.uniform(0.95, 1.05), 1)
        fat  = round(item[5] * np.random.uniform(0.95, 1.05), 1)
        sod  = round(item[6] * np.random.uniform(0.95, 1.05), 1)
        fib  = round(item[7] * np.random.uniform(0.95, 1.05), 2)
        satf = round(item[8] * np.random.uniform(0.95, 1.05), 2)
        rows.append([item[0], item[1], cal, carb, prot, fat, sod, fib, satf])

    df = pd.DataFrame(rows, columns=FOOD_COLUMNS)
    print(f"[OK] food_nutrition.csv: {df.shape[0]} item makanan")
    print(f"     Distribusi Meal_Type:\n{df['Meal_Type'].value_counts().to_string()}")
    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Generate Datasets untuk DSS Diet")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Dataset profil pasien
    df_diet = generate_diet_recommendations(N_PATIENTS)
    path_diet = OUTPUT_DIR / "diet_recommendations.csv"
    df_diet.to_csv(path_diet, index=False)
    print(f"     Tersimpan: {path_diet}\n")

    # 2. Database makanan
    df_food = generate_food_nutrition()
    path_food = OUTPUT_DIR / "food_nutrition.csv"
    df_food.to_csv(path_food, index=False)
    print(f"     Tersimpan: {path_food}\n")

    print("=" * 55)
    print("  [SELESAI] Dataset berhasil dibuat.")
    print("  Langkah berikutnya: python src/train.py")
    print("=" * 55)
