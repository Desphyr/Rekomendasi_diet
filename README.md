# Diet DSS — Sistem Pendukung Keputusan Rekomendasi Menu Diet Harian

> Hybrid AI + Rule-Based Decision Support System menggunakan Random Forest & Nutrisi-Rule Filtering

---

## 🗂 Struktur Proyek

```
Program diet/
├── data/
│   ├── generate_datasets.py      # Script generate dataset sintetis
│   ├── diet_recommendations.csv  # Dataset training RF (auto-generated)
│   └── food_nutrition.csv        # Database makanan (auto-generated)
│
├── models/
│   └── rf_diet_model.pkl         # Model RF + preprocessor (auto-generated)
│
├── src/
│   ├── train.py                  # Training pipeline Random Forest
│   ├── predictor.py              # Wrapper inferensi model RF
│   ├── recommender.py            # DietRecommender (Rule-Based + Scoring)
│   └── schemas.py                # Pydantic schemas (validasi I/O)
│
├── app.py                        # FastAPI REST API
├── demo.py                       # CLI demo (tanpa server)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Instalasi

```bash
# 1. Buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset sintetis
python data/generate_datasets.py

# 4. Latih model Random Forest
python src/train.py

# 5a. Jalankan CLI demo
python demo.py

# 5b. ATAU jalankan REST API
uvicorn app:app --reload
# → Buka http://localhost:8000/docs untuk Swagger UI
```

---

## 🔄 Alur Kerja Sistem

```
Input Profil Pengguna
        │
        ▼
┌─────────────────────┐
│ Modul 1: Validasi   │  Pydantic (schemas.py)
│ Input & BMI Calc    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Modul 2: Klasifikasi│  Random Forest (predictor.py)
│ Random Forest       │  → Balanced | Low_Carb | Low_Sodium
│ + Probabilitas      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│ Modul 3: Rule-Based Filtering (recommender.py)  │
│  Hard Constraint → Tolak item/menu melanggar    │
│  Soft Constraint → Skor kedekatan ke target     │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│ Modul 4: Pembentukan│  Kombinasi Breakfast+Lunch+
│ Menu Harian         │  Dinner+Snack
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Modul 5: Perankingan│  Skor gabungan 6 komponen
│ & Scoring           │  nutrisi berbobot
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Modul 6: Explain-   │  Teks alasan per menu +
│ ability             │  penjelasan global
└─────────┬───────────┘
          │
          ▼
      Output DSS
```

---

## 📊 Model Random Forest

| Parameter           | Nilai             |
|---------------------|-------------------|
| n_estimators        | 500               |
| max_features        | sqrt              |
| class_weight        | balanced          |
| Cross-Validation    | 5-Fold Stratified |
| Target Classes      | 3 (Balanced, Low_Carb, Low_Sodium) |

### Fitur Input Model

| Fitur Input (Total 20)          | Tipe      | Preprocessing     |
|---------------------------------|-----------|-------------------|
| Age                             | Numerik   | StandardScaler    |
| Weight_kg                       | Numerik   | StandardScaler    |
| Height_cm                       | Numerik   | StandardScaler    |
| BMI                             | Numerik   | StandardScaler    |
| Glucose_mg_dL                   | Numerik   | StandardScaler    |
| Blood_Pressure_mmHg             | Numerik   | StandardScaler    |
| Cholesterol_mg_dL               | Numerik   | StandardScaler    |
| Daily_Caloric_Intake            | Numerik   | StandardScaler    |
| Weekly_Exercise_Hours           | Numerik   | StandardScaler    |
| Adherence_to_Diet_Plan          | Numerik   | StandardScaler    |
| Dietary_Nutrient_Imbalance_Score| Numerik   | StandardScaler    |
| Activity_Level                  | Ordinal   | OrdinalEncoder    |
| Severity                        | Ordinal   | OrdinalEncoder    |
| Gender                          | Biner     | Manual (0/1)      |
| Disease_Type                    | Kategori  | One-Hot Encoding  |
| Dietary_Restrictions            | Kategori  | One-Hot Encoding  |
| Preferred_Cuisine               | Kategori  | One-Hot Encoding  |

*(Sesuai laporan penelitian, fitur alergi tidak digunakan karena merupakan batasan sistem, dan **Ablation Study** otomatis dilakukan pada atribut `Disease_Type` saat training model).*

---

## 🥗 Aturan Nutrisi per Tipe Diet

| Nutrisi           | Balanced        | Low_Carb        | Low_Sodium      |
|-------------------|-----------------|-----------------|-----------------|
| Kalori (kcal/hr)  | 1600–2200       | 1400–1900       | 1600–2200       |
| Karbohidrat (%)   | 45–65%          | **5–25%**       | 45–65%          |
| Protein (%)       | 10–35%          | 25–45%          | 10–35%          |
| Lemak (%)         | 20–35%          | 30–60%          | 20–35%          |
| Sodium maks (mg)  | 2300            | 2300            | **1500**        |
| Serat min (g)     | 25              | 20              | 25              |
| Lemak Jenuh (g)   | ≤20             | ≤20             | ≤18             |

---

## 🌐 REST API Endpoints

| Method | Endpoint       | Deskripsi                       |
|--------|----------------|---------------------------------|
| GET    | `/health`      | Health check sistem             |
| GET    | `/diet-rules`  | Tampilkan aturan nutrisi        |
| POST   | `/recommend`   | **Rekomendasi menu diet utama** |

### Contoh Request `/recommend`

```json
{
  "Age": 52,
  "Gender": "Male",
  "Weight_kg": 85,
  "Height_cm": 170,
  "Activity_Level": "Light",
  "Weekly_Exercise_Hours": 2.0,
  "Disease_Type": "Diabetes",
  "Severity": "Moderate",
  "Glucose_mg_dL": 185,
  "Blood_Pressure_mmHg": 128,
  "Cholesterol_mg_dL": 210,
  "Daily_Caloric_Intake": 2400.0,
  "Adherence_to_Diet_Plan": 55.0,
  "Dietary_Nutrient_Imbalance_Score": 4.5,
  "Dietary_Restrictions": "Low_Sugar",
  "Preferred_Cuisine": "None",
  "top_n_menus": 3
}
```

### Contoh Response

```json
{
  "status": "success",
  "diet_type": "Low_Carb",
  "diet_probability": {"Balanced": 0.05, "Low_Carb": 0.91, "Low_Sodium": 0.04},
  "diet_description": "Menu rendah karbohidrat...",
  "target_calories_kcal": 1875,
  "recommended_menus": [ ... ],
  "compliance_status": "COMPLIANT — Menu memenuhi semua aturan diet.",
  "global_explanation": [ ... ]
}
```

---

## ⚠️ Disclaimer

Sistem ini adalah **alat pendukung keputusan (DSS)**, bukan pengganti diagnosis atau saran medis profesional. Selalu konsultasikan rekomendasi diet dengan dokter atau ahli gizi bersertifikat.
