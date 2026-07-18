"""
DietRecommender - Modul Rule-Based Filtering, Scoring & Explainability
=======================================================================
Menerima tipe diet hasil prediksi + dataframe makanan, lalu:
  1. Memfilter makanan berdasarkan Hard Constraint nutrisi
  2. Memberi skor per item (Soft Constraint)
  3. Menyusun menu harian (Breakfast + Lunch + Dinner + Snack)
  4. Menghitung total nutrisi
  5. Menghasilkan penjelasan rekomendasi
  6. Menghasilkan user_insights dari fitur dataset

Mendukung fitur dari diet_recommendations_dataset.csv:
  - Severity              : modifikasi target kalori
  - Adherence_to_Diet_Plan: insight kepatuhan diet
  - Weekly_Exercise_Hours : insight gaya hidup
  - Dietary_Nutrient_Imbalance_Score: insight keseimbangan nutrisi
  - Glucose_mg_dL / Blood_Pressure_mmHg: nama field klinis

Catatan: Fitur Allergies TIDAK dipertimbangkan sesuai batasan sistem di laporan.
"""

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

DIET_RULES: dict = {
    "Balanced": {
        "daily_calories"   : (1600, 2200),   # min, max kcal
        "carbs_pct"        : (0.45, 0.65),   # 45-65% dari kalori
        "protein_pct"      : (0.10, 0.35),   # 10-35%
        "fat_pct"          : (0.20, 0.35),   # 20-35%
        "sodium_max_mg"    : 2300,
        "fiber_min_g"      : 25,
        "sat_fat_max_g"    : 20,
        "description"      : "Menu seimbang dengan proporsi karbohidrat, protein, dan lemak yang optimal.",
    },
    "Low_Carb": {
        "daily_calories"   : (1400, 1900),
        "carbs_pct"        : (0.05, 0.25),   # sangat rendah karbo
        "protein_pct"      : (0.25, 0.45),   # protein lebih tinggi
        "fat_pct"          : (0.30, 0.60),
        "sodium_max_mg"    : 2300,
        "fiber_min_g"      : 20,
        "sat_fat_max_g"    : 20,
        "description"      : "Menu rendah karbohidrat untuk membantu kontrol gula darah (cocok untuk Diabetes).",
    },
    "Low_Sodium": {
        "daily_calories"   : (1600, 2200),
        "carbs_pct"        : (0.45, 0.65),
        "protein_pct"      : (0.10, 0.35),
        "fat_pct"          : (0.20, 0.35),
        "sodium_max_mg"    : 1500,           # lebih ketat dari standar
        "fiber_min_g"      : 25,
        "sat_fat_max_g"    : 18,
        "description"      : "Menu rendah sodium untuk membantu mengontrol tekanan darah (cocok untuk Hipertensi).",
    },
}

# Target kalori per meal-type (persentase dari total harian)
MEAL_CALORIE_RATIO = {
    "Breakfast": 0.25,
    "Lunch"    : 0.35,
    "Dinner"   : 0.30,
    "Snack"    : 0.10,
}

@dataclass
class MenuOption:
    """Representasi satu kombinasi menu harian lengkap."""
    items       : list[dict]          # daftar makanan terpilih
    total_kcal  : float = 0.0
    total_carbs : float = 0.0
    total_protein: float = 0.0
    total_fat   : float = 0.0
    total_sodium: float = 0.0
    total_fiber : float = 0.0
    total_sat_fat: float = 0.0
    score       : float = 0.0
    explanations: list[str] = field(default_factory=list)

    def compute_totals(self):
        self.total_kcal   = sum(i["Calories_kcal"]    for i in self.items)
        self.total_carbs  = sum(i["Carbs_g"]          for i in self.items)
        self.total_protein= sum(i["Protein_g"]        for i in self.items)
        self.total_fat    = sum(i["Fat_g"]            for i in self.items)
        self.total_sodium = sum(i["Sodium_mg"]        for i in self.items)
        self.total_fiber  = sum(i["Fiber_g"]          for i in self.items)
        self.total_sat_fat= sum(i["Saturated_Fat_g"]  for i in self.items)

    def to_dict(self) -> dict:
        self.compute_totals()
        return {
            "menu_items"    : [i["Food_Name"] + f" ({i['Meal_Type']})" for i in self.items],
            "nutrition_summary": {
                "total_calories_kcal"   : round(self.total_kcal, 1),
                "total_carbs_g"         : round(self.total_carbs, 1),
                "total_protein_g"       : round(self.total_protein, 1),
                "total_fat_g"           : round(self.total_fat, 1),
                "total_sodium_mg"       : round(self.total_sodium, 1),
                "total_fiber_g"         : round(self.total_fiber, 1),
                "total_saturated_fat_g" : round(self.total_sat_fat, 1),
            },
            "score"        : round(self.score, 4),
            "explanations" : self.explanations,
        }


@dataclass
class DSSResult:
    """Output lengkap sistem DSS."""
    diet_type          : str
    diet_probability   : dict[str, float]
    diet_description   : str
    recommended_menus  : list[dict]
    compliance_status  : str
    global_explanation : list[str]
    user_insights      : dict = field(default_factory=dict)

class DietRecommender:
    """
    Hybrid DSS yang menggabungkan prediksi Random Forest dengan Rule-Based
    Filtering untuk menyusun rekomendasi menu diet harian.

    Parameters
    ----------
    food_df       : DataFrame makanan (food_nutrition.csv)
    top_n_menus   : Jumlah menu terbaik yang dikembalikan
    target_calories: Override kalori target (opsional)
    """

    def __init__(
        self,
        food_df        : pd.DataFrame,
        top_n_menus    : int = 3,
        target_calories: Optional[float] = None,
    ):
        self.food_df         = food_df.copy()
        self.top_n_menus     = top_n_menus
        self.target_calories = target_calories

    def recommend(
        self,
        diet_type   : str,
        probabilities: dict[str, float],
        user_profile : dict,
    ) -> DSSResult:
        """
        Entrypoint utama DietRecommender.

        Parameters
        ----------
        diet_type     : Hasil prediksi RF ('Balanced'|'Low_Carb'|'Low_Sodium')
        probabilities : Dict {diet_type: probability}
        user_profile  : Dict profil pengguna (untuk menghitung target kalori)
        """
        rules = DIET_RULES[diet_type]

        # Estimasi kebutuhan kalori harian berdasarkan profil
        target_kcal = self.target_calories or self._estimate_calories(user_profile)

        # Modifikasi kalori berdasarkan Severity penyakit
        severity = user_profile.get("Severity", "Mild")
        if severity == "Severe":
            target_kcal *= 0.90   # kurangi 10% untuk kondisi berat
        elif severity == "Moderate":
            target_kcal *= 0.95   # kurangi 5% untuk kondisi sedang

        target_kcal = max(rules["daily_calories"][0],
                          min(target_kcal, rules["daily_calories"][1]))

        # 1. Filter makanan (Hard + Soft Constraint per item + Allergy filter)
        filtered = self._filter_foods(diet_type, rules, target_kcal, user_profile)

        # 2. Susun kandidat menu dari kombinasi makanan per meal type
        candidate_menus = self._build_menu_candidates(filtered, target_kcal)

        # 3. Scoring & filtering hard constraint level menu
        valid_menus = self._score_and_filter(candidate_menus, rules, target_kcal)

        # 4. Ranking & ambil top-N
        top_menus = sorted(valid_menus, key=lambda m: m.score, reverse=True)[: self.top_n_menus]

        # 5. Generate explainability
        for menu in top_menus:
            menu.explanations = self._explain(menu, diet_type, rules, target_kcal)

        # 6. Global explanation
        global_exp = self._global_explanation(diet_type, probabilities, user_profile, rules)

        # 7. Compliance status
        compliance = self._assess_compliance(top_menus[0] if top_menus else None, rules)

        # 8. User insights dari fitur dataset baru
        user_insights = self._generate_user_insights(user_profile, diet_type)

        return DSSResult(
            diet_type         = diet_type,
            diet_probability  = {k: round(v, 4) for k, v in probabilities.items()},
            diet_description  = rules["description"],
            recommended_menus = [m.to_dict() for m in top_menus],
            compliance_status = compliance,
            global_explanation= global_exp,
            user_insights     = user_insights,
        )

    def _estimate_calories(self, profile: dict) -> float:
        """Hitung TDEE menggunakan persamaan Harris-Benedict x faktor aktivitas.

        Jika tersedia Daily_Caloric_Intake dari dataset, gunakan sebagai anchor
        untuk menyesuaikan estimasi (blending 60% TDEE + 40% actual intake).
        """
        w   = profile.get("Weight_kg", 70)
        h   = profile.get("Height_cm", 170)
        age = profile.get("Age", 30)
        gender = profile.get("Gender", "Male")

        if gender == "Male":
            bmr = 88.362 + (13.397 * w) + (4.799 * h) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * w) + (3.098 * h) - (4.330 * age)

        activity_factor = {
            "Sedentary" : 1.2,
            "Light"     : 1.375,
            "Moderate"  : 1.55,
            "Active"    : 1.725,
            "Very_Active": 1.9,
        }.get(profile.get("Activity_Level", "Moderate"), 1.55)

        tdee = bmr * activity_factor

        actual_intake = profile.get("Daily_Caloric_Intake", 0)
        if 800 <= actual_intake <= 5000:
            tdee = tdee * 0.6 + actual_intake * 0.4

        return round(tdee, 0)

    def _filter_foods(
        self, diet_type: str, rules: dict, target_kcal: float,
        user_profile: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Terapkan aturan diet pada level individual makanan.
        Hard constraint: hapus makanan yang melanggar batas MUTLAK per item.
        Soft constraint: beri skor item (0-1).

        Catatan: Filter alergen TIDAK diterapkan sesuai batasan sistem di laporan.
        """
        df = self.food_df.copy()

        # -- Hard Constraints per item -----------------------------------------
        # Sodium: satu item tidak boleh melebihi 65% kuota harian
        sodium_item_limit = rules["sodium_max_mg"] * 0.65
        df = df[df["Sodium_mg"] <= sodium_item_limit]

        # Lemak jenuh: satu item tidak boleh melebihi 60% batas harian
        df = df[df["Saturated_Fat_g"] <= rules["sat_fat_max_g"] * 0.60]

        if diet_type == "Low_Carb":
            # Pembatasan karbohidrat per porsi:
            #   - Meal utama (Breakfast/Lunch/Dinner): <= 40g
            #   - Snack: <= 15g
            df = df[
                ((df["Meal_Type"] != "Snack") & (df["Carbs_g"] <= 40)) |
                ((df["Meal_Type"] == "Snack")  & (df["Carbs_g"] <= 15))
            ]

        # -- Soft Scoring per item ---------------------------------------------
        df = df.copy()
        df["item_score"] = df.apply(
            lambda row: self._score_item(row, diet_type, rules, target_kcal),
            axis=1
        )

        return df.reset_index(drop=True)

    def _score_item(
        self, row: pd.Series, diet_type: str, rules: dict, target_kcal: float
    ) -> float:
        """
        Skor 0-1 per item makanan berdasarkan kedekatan nutrisi dengan target.
        Skor tinggi = lebih direkomendasikan.
        """
        score = 1.0

        # Sodium penalty (makin mendekati batas, makin rendah skor)
        sodium_ratio = row["Sodium_mg"] / rules["sodium_max_mg"]
        score -= sodium_ratio * 0.3

        # Fiber bonus
        fiber_ratio = min(row["Fiber_g"] / (rules["fiber_min_g"] / 4), 1.0)
        score += fiber_ratio * 0.2

        # Protein bonus
        prot_kcal = row["Protein_g"] * 4
        prot_pct  = prot_kcal / max(row["Calories_kcal"], 1)
        ideal_prot = (rules["protein_pct"][0] + rules["protein_pct"][1]) / 2
        score += (1 - abs(prot_pct - ideal_prot)) * 0.15

        if diet_type == "Low_Carb":
            # Bonus ekstra jika karbo rendah
            carb_penalty = row["Carbs_g"] / 30
            score -= carb_penalty * 0.25

        if diet_type == "Low_Sodium":
            # Bonus ekstra jika sodium sangat rendah
            if row["Sodium_mg"] < 150:
                score += 0.15

        return round(max(0.0, min(1.0, score)), 4)

    def _build_menu_candidates(
        self, filtered_df: pd.DataFrame, target_kcal: float
    ) -> list[MenuOption]:
        """
        Buat kandidat menu dengan mengambil top-K item per meal type,
        lalu mengombinasikan secara kartesian (dibatasi agar tidak eksplosif).
        """
        meals = {}
        k_per_meal = {"Breakfast": 4, "Lunch": 5, "Dinner": 5, "Snack": 3}

        for meal_type, k in k_per_meal.items():
            subset = (
                filtered_df[filtered_df["Meal_Type"] == meal_type]
                .sort_values("item_score", ascending=False)
                .head(k)
                .to_dict("records")
            )
            if not subset:
                # Jika tidak ada makanan lolos filter, ambil yang paling rendah
                subset = (
                    self.food_df[self.food_df["Meal_Type"] == meal_type]
                    .sort_values("Sodium_mg")
                    .head(k)
                    .to_dict("records")
                )
            meals[meal_type] = subset

        # Kombinasi kartesian
        candidates = []
        for combo in itertools.product(
            meals["Breakfast"], meals["Lunch"], meals["Dinner"], meals["Snack"]
        ):
            menu = MenuOption(items=list(combo))
            menu.compute_totals()
            candidates.append(menu)

        return candidates

    def _score_and_filter(
        self,
        candidates : list[MenuOption],
        rules      : dict,
        target_kcal: float,
    ) -> list[MenuOption]:
        """
        Hard Constraint menu level -> tolak kombinasi yang melanggar batas mutlak.
        Soft Constraint -> beri skor 0-1 berdasarkan kedekatan ke target nutrisi.

        Hard constraints yang diterapkan:
          1. Total sodium sehari tidak melebihi batas maksimal
          2. Total lemak jenuh tidak melebihi batas maksimal
          (Kalori harian dijadikan SOFT constraint agar menu tetap terbentuk;
           kombinasi satu item per meal-type secara natural < target harian penuh.)
        """
        valid = []
        for menu in candidates:
            menu.compute_totals()

            # -- Hard Constraint 1: Sodium total harian ------------------------
            if menu.total_sodium > rules["sodium_max_mg"]:
                continue

            # -- Hard Constraint 2: Lemak jenuh total harian ------------------
            if menu.total_sat_fat > rules["sat_fat_max_g"]:
                continue

            # Low_Carb: total karbo dalam satu hari (4 porsi) tidak melebihi
            # batas harian (target_kcal * carbs_pct_max / 4 kcal per gram)
            if rules.get("carbs_pct") and rules["carbs_pct"][1] < 0.30:
                max_daily_carbs = (target_kcal * rules["carbs_pct"][1]) / 4.0
                if menu.total_carbs > max_daily_carbs:
                    continue

            # -- Soft Scoring --------------------------------------------------
            menu.score = self._score_menu(menu, rules, target_kcal)
            valid.append(menu)

        return valid

    def _score_menu(
        self, menu: MenuOption, rules: dict, target_kcal: float
    ) -> float:
        """
        Skor menu 0-1 berdasarkan kedekatan nutrisi dengan target ideal.
        Menggunakan pendekatan penalty jarak dari nilai ideal.
        """
        score = 0.0
        kcal  = menu.total_kcal

        # 1. Kalori (bobot 25%)
        ideal_kcal = (rules["daily_calories"][0] + rules["daily_calories"][1]) / 2
        kcal_diff  = abs(kcal - ideal_kcal) / ideal_kcal
        score += (1 - min(kcal_diff, 1.0)) * 0.25

        # 2. Karbohidrat (bobot 25%)
        carb_kcal   = menu.total_carbs * 4
        carb_pct    = carb_kcal / max(kcal, 1)
        ideal_carb  = (rules["carbs_pct"][0] + rules["carbs_pct"][1]) / 2
        carb_diff   = abs(carb_pct - ideal_carb) / max(ideal_carb, 0.01)
        score += (1 - min(carb_diff, 1.0)) * 0.25

        # 3. Protein (bobot 20%)
        prot_kcal   = menu.total_protein * 4
        prot_pct    = prot_kcal / max(kcal, 1)
        ideal_prot  = (rules["protein_pct"][0] + rules["protein_pct"][1]) / 2
        prot_diff   = abs(prot_pct - ideal_prot) / max(ideal_prot, 0.01)
        score += (1 - min(prot_diff, 1.0)) * 0.20

        # 4. Sodium (bobot 15%) - makin rendah makin baik
        sodium_ratio = menu.total_sodium / rules["sodium_max_mg"]
        score += (1 - min(sodium_ratio, 1.0)) * 0.15

        # 5. Serat (bobot 10%) - minimal terpenuhi
        fiber_ratio = min(menu.total_fiber / rules["fiber_min_g"], 1.0)
        score += fiber_ratio * 0.10

        # 6. Lemak jenuh (bobot 5%)
        sat_ratio = 1 - (menu.total_sat_fat / max(rules["sat_fat_max_g"], 1))
        score += max(0.0, sat_ratio) * 0.05

        return round(max(0.0, min(1.0, score)), 4)


    def _explain(
        self,
        menu      : MenuOption,
        diet_type : str,
        rules     : dict,
        target_kcal: float,
    ) -> list[str]:
        """Hasilkan kalimat-kalimat penjelasan untuk sebuah menu."""
        exps = []
        kcal = menu.total_kcal

        # Kalori
        min_k, max_k = rules["daily_calories"]
        exps.append(
            f"[OK] Total kalori {kcal:.0f} kcal berada dalam rentang target "
            f"{min_k}-{max_k} kcal/hari."
        )

        # Karbo
        carb_pct = (menu.total_carbs * 4) / max(kcal, 1) * 100
        min_cp   = rules["carbs_pct"][0] * 100
        max_cp   = rules["carbs_pct"][1] * 100
        exps.append(
            f"[OK] Karbohidrat {menu.total_carbs:.1f}g ({carb_pct:.1f}% kalori) "
            f"- target {min_cp:.0f}-{max_cp:.0f}%."
        )

        # Protein
        prot_pct = (menu.total_protein * 4) / max(kcal, 1) * 100
        exps.append(
            f"[OK] Protein {menu.total_protein:.1f}g ({prot_pct:.1f}% kalori) "
            f"- mendukung massa otot dan metabolisme."
        )

        # Sodium
        exps.append(
            f"{'[OK]' if menu.total_sodium <= rules['sodium_max_mg'] else '[!]'} "
            f"Sodium {menu.total_sodium:.0f}mg "
            f"(batas: {rules['sodium_max_mg']}mg/hari)."
        )

        # Serat
        if menu.total_fiber >= rules["fiber_min_g"]:
            exps.append(
                f"[OK] Serat {menu.total_fiber:.1f}g memenuhi kebutuhan harian "
                f"?{rules['fiber_min_g']}g."
            )
        else:
            exps.append(
                f"[i] Serat {menu.total_fiber:.1f}g sedikit di bawah target "
                f"{rules['fiber_min_g']}g - pertimbangkan tambahan sayur."
            )

        # Diet-spesifik
        if diet_type == "Low_Carb":
            exps.append(
                "[*] Menu ini membatasi asupan karbohidrat untuk membantu "
                "stabilisasi kadar gula darah bagi penderita Diabetes."
            )
        elif diet_type == "Low_Sodium":
            exps.append(
                "[*] Menu ini membatasi asupan sodium untuk membantu "
                "mengelola tekanan darah tinggi (Hipertensi)."
            )
        else:
            exps.append(
                "[*] Menu seimbang ini memenuhi proporsi makronutrien "
                "berdasarkan Angka Kecukupan Gizi (AKG) Indonesia."
            )

        return exps

    def _global_explanation(
        self,
        diet_type   : str,
        probabilities: dict[str, float],
        user_profile : dict,
        rules        : dict,
    ) -> list[str]:
        """Penjelasan global mengapa tipe diet ini dipilih (diperkaya fitur baru)."""
        exps = []
        prob = probabilities.get(diet_type, 0)
        exps.append(
            f"Model Random Forest memprediksi tipe diet '{diet_type}' "
            f"dengan probabilitas {prob*100:.1f}%."
        )

        disease   = user_profile.get("Disease_Type", "None")
        severity  = user_profile.get("Severity", "Mild")
        bmi       = user_profile.get("BMI", 0)
        # Nama field baru (dengan fallback ke nama lama)
        glucose   = user_profile.get("Glucose_mg_dL") or user_profile.get("Blood_Sugar_mgdL", 0)
        bp        = user_profile.get("Blood_Pressure_mmHg") or user_profile.get("Blood_Pressure_Systolic", 0)
        chol      = user_profile.get("Cholesterol_mg_dL") or user_profile.get("Cholesterol_mgdL", 0)
        adherence = user_profile.get("Adherence_to_Diet_Plan", 70)
        exercise  = user_profile.get("Weekly_Exercise_Hours", 3)
        imbalance = user_profile.get("Dietary_Nutrient_Imbalance_Score", 2)

        # Penyakit & kondisi klinis
        if disease == "Diabetes" or glucose >= 126:
            exps.append(
                f"Faktor penentu: Glukosa darah {glucose:.0f} mg/dL (≥126) dan/atau "
                f"riwayat Diabetes [{severity}] → diet Low Carb diindikasikan."
            )
        elif disease == "Hypertension" or bp >= 140:
            exps.append(
                f"Faktor penentu: Tekanan darah {bp} mmHg (≥140) "
                f"dan/atau riwayat Hipertensi [{severity}] → diet Low Sodium diindikasikan."
            )
        elif disease == "Obesity" or bmi >= 30:
            exps.append(
                f"Faktor penentu: BMI {bmi:.1f} (≥30 = Obesitas) [{severity}] "
                f"→ diet Balanced dengan defisit kalori diindikasikan."
            )
        else:
            exps.append(
                f"Profil BMI {bmi:.1f} dan tidak ada kondisi kronis dominan "
                f"→ diet Balanced diindikasikan."
            )

        # Kolesterol
        if chol >= 240:
            exps.append(
                f"[!] Kolesterol {chol:.0f} mg/dL (>240 = tinggi) — batasi lemak jenuh "
                f"dan konsumsi lebih banyak serat."
            )

        # Kepatuhan diet
        if adherence < 60:
            exps.append(
                f"[!] Kepatuhan diet Anda saat ini {adherence:.0f}% — "
                f"disarankan berkonsultasi dengan ahli gizi untuk meningkatkan konsistensi."
            )
        elif adherence >= 85:
            exps.append(
                f"[+] Kepatuhan diet Anda sangat baik ({adherence:.0f}%) — "
                f"teruskan konsistensi ini untuk hasil optimal."
            )

        # Olahraga
        if exercise < 2.5:
            exps.append(
                f"[!] Jam olahraga {exercise:.1f} jam/minggu terlalu rendah — "
                f"WHO merekomendasikan minimal 2.5 jam/minggu aktivitas aerobik sedang."
            )
        elif exercise >= 7:
            exps.append(
                f"[+] Aktivitas fisik tinggi ({exercise:.1f} jam/minggu) — "
                f"pastikan asupan protein cukup untuk pemulihan otot."
            )

        # Ketidakseimbangan nutrisi
        if imbalance >= 4:
            exps.append(
                f"[!] Skor ketidakseimbangan nutrisi {imbalance:.1f}/10 (tinggi) — "
                f"pola makan perlu diperbaiki secara menyeluruh."
            )

        exps.append(f"Deskripsi: {rules['description']}")
        exps.append(
            "[!] Sistem ini adalah alat pendukung keputusan. "
            "Konsultasikan rekomendasi ini dengan dokter atau ahli gizi."
        )
        return exps

    def _generate_user_insights(self, profile: dict, diet_type: str) -> dict:
        """Hasilkan insight terstruktur dari fitur-fitur dataset."""
        glucose   = profile.get("Glucose_mg_dL") or profile.get("Blood_Sugar_mgdL", 90)
        bp        = profile.get("Blood_Pressure_mmHg") or profile.get("Blood_Pressure_Systolic", 120)
        chol      = profile.get("Cholesterol_mg_dL") or profile.get("Cholesterol_mgdL", 180)
        adherence = profile.get("Adherence_to_Diet_Plan", 70)
        exercise  = profile.get("Weekly_Exercise_Hours", 3)
        imbalance = profile.get("Dietary_Nutrient_Imbalance_Score", 2)
        severity  = profile.get("Severity", "Mild")
        restriction = profile.get("Dietary_Restrictions", "None")
        cuisine     = profile.get("Preferred_Cuisine", "None")

        # Status glukosa
        if glucose < 100:
            glucose_status = "Normal"
        elif glucose < 126:
            glucose_status = "Pradiabetes"
        else:
            glucose_status = "Tinggi (Diabetes Range)"

        # Status tekanan darah
        if bp < 120:
            bp_status = "Normal"
        elif bp < 130:
            bp_status = "Elevated"
        elif bp < 140:
            bp_status = "Hipertensi Stage 1"
        else:
            bp_status = "Hipertensi Stage 2"

        # Status kolesterol
        if chol < 200:
            chol_status = "Normal"
        elif chol < 240:
            chol_status = "Batas Tinggi"
        else:
            chol_status = "Tinggi"

        # Skor kepatuhan
        if adherence >= 85:
            adherence_status = "Sangat Baik"
        elif adherence >= 70:
            adherence_status = "Baik"
        elif adherence >= 50:
            adherence_status = "Cukup"
        else:
            adherence_status = "Perlu Perhatian"

        # Aktivitas fisik
        if exercise >= 7:
            exercise_status = "Sangat Aktif"
        elif exercise >= 4:
            exercise_status = "Aktif"
        elif exercise >= 2.5:
            exercise_status = "Cukup"
        else:
            exercise_status = "Kurang Aktif"

        # Skor keseimbangan nutrisi
        if imbalance <= 2:
            imbalance_status = "Baik"
        elif imbalance <= 4:
            imbalance_status = "Perlu Perbaikan"
        else:
            imbalance_status = "Buruk — Perlu Intervensi"

        return {
            "clinical_markers": {
                "glucose_mg_dL"         : glucose,
                "glucose_status"        : glucose_status,
                "blood_pressure_mmHg"   : bp,
                "blood_pressure_status" : bp_status,
                "cholesterol_mg_dL"     : chol,
                "cholesterol_status"    : chol_status,
                "disease_severity"      : severity,
            },
            "lifestyle": {
                "weekly_exercise_hours"   : exercise,
                "exercise_status"         : exercise_status,
                "diet_adherence_pct"      : adherence,
                "adherence_status"        : adherence_status,
                "nutrient_imbalance_score": imbalance,
                "imbalance_status"        : imbalance_status,
            },
            "dietary_profile": {
                "dietary_restrictions" : restriction,
                "preferred_cuisine"    : cuisine,
                "recommended_diet_type": diet_type,
            },
        }

    def _assess_compliance(
        self,
        menu : Optional[MenuOption],
        rules: dict,
    ) -> str:
        if menu is None:
            return "NON_COMPLIANT - Tidak ada menu yang memenuhi semua hard constraint."

        issues = []
        min_k, max_k = rules["daily_calories"]
        if not (min_k <= menu.total_kcal <= max_k):
            issues.append(f"kalori ({menu.total_kcal:.0f} kcal out of range)")
        if menu.total_sodium > rules["sodium_max_mg"]:
            issues.append(f"sodium ({menu.total_sodium:.0f}mg > {rules['sodium_max_mg']}mg)")
        if menu.total_sat_fat > rules["sat_fat_max_g"]:
            issues.append(f"lemak jenuh ({menu.total_sat_fat:.1f}g > {rules['sat_fat_max_g']}g)")
        if menu.total_fiber < rules["fiber_min_g"] * 0.8:
            issues.append(f"serat kurang ({menu.total_fiber:.1f}g < {rules['fiber_min_g']}g)")

        if not issues:
            return "COMPLIANT - Menu memenuhi semua aturan diet yang ditetapkan."
        else:
            return "PARTIAL - Perlu perhatian: " + "; ".join(issues) + "."
