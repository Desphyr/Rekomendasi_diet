"""
FastAPI Application — Diet DSS REST API
========================================
Endpoint:
  POST /recommend  → Terima profil pengguna, kembalikan rekomendasi menu diet
  GET  /health     → Health check
  GET  /diet-rules → Tampilkan semua aturan diet yang digunakan
"""

import os
import socket
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.schemas    import UserProfile, DSSResponse
from src.predictor  import DietPredictor
from src.recommender import DietRecommender, DIET_RULES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"


def get_available_port(start_port: int = 8000, host: str = "127.0.0.1") -> int:
    """Return a free port, falling back to the next available one."""
    preferred_port = int(os.getenv("PORT", str(start_port)))

    for candidate in [preferred_port, *range(preferred_port + 1, preferred_port + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue

    raise RuntimeError("Tidak ada port yang tersedia untuk menjalankan server.")

app = FastAPI(
    title       = "Diet DSS — Sistem Pendukung Keputusan Rekomendasi Menu Diet",
    description = (
        "Sistem hybrid berbasis Random Forest + Rule-Based Filtering "
        "untuk merekomendasikan menu diet harian yang optimal berdasarkan "
        "profil kesehatan pengguna."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder (opsional — hanya dipakai jika Anda masih ingin
# menyajikan halaman statis langsung dari FastAPI di Render)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", tags=["System"])
def serve_ui():
    """Serve halaman statis (opsional, tidak dipakai oleh Streamlit)."""
    return FileResponse(str(STATIC_DIR / "index.html"))

try:
    predictor = DietPredictor()
    food_path = DATA_DIR / "food_nutrition.csv"
    food_df   = pd.read_csv(food_path)
    print("[OK] Model dan database makanan berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[!] Warning: {e}")
    predictor = None
    food_df   = None


def start_server(host: str = "0.0.0.0", port: int | None = None):
    """Jalankan API secara lokal (untuk development, bukan dipakai di Render)."""
    import uvicorn

    resolved_port = get_available_port(start_port=port or 8000, host="127.0.0.1")
    print(f"[INFO] Menjalankan API di http://127.0.0.1:{resolved_port}")
    uvicorn.run(app, host=host, port=resolved_port, reload=False)

@app.get("/health", tags=["System"])
def health_check():
    """Cek status sistem."""
    return {
        "status"       : "OK",
        "model_loaded" : predictor is not None,
        "food_db_loaded": food_df is not None and len(food_df) > 0,
    }


@app.get("/diet-rules", tags=["System"])
def get_diet_rules():
    """Kembalikan semua aturan nutrisi yang digunakan sistem."""
    return {"diet_rules": DIET_RULES}


@app.post("/recommend", response_model=DSSResponse, tags=["Recommendation"])
def recommend(profile: UserProfile):
    """
    Endpoint utama: Terima profil pengguna, kembalikan rekomendasi menu diet.

    **Workflow:**
    1. Validasi input (Pydantic)
    2. Prediksi tipe diet (Random Forest)
    3. Filter & scoring makanan (Rule-Based)
    4. Susun & ranking menu harian
    5. Hasilkan penjelasan
    """
    if predictor is None or food_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum dilatih. Jalankan 'python src/train.py' terlebih dahulu."
        )

    user_dict = profile.to_dict()

    diet_type, probabilities = predictor.predict(user_dict)

    recommender = DietRecommender(
        food_df        = food_df,
        top_n_menus    = profile.top_n_menus,
        target_calories= profile.target_calories,
    )
    result = recommender.recommend(diet_type, probabilities, user_dict)

    target_kcal = profile.target_calories or recommender._estimate_calories(user_dict)

    return DSSResponse(
        status              = "success",
        diet_type           = result.diet_type,
        diet_probability    = result.diet_probability,
        diet_description    = result.diet_description,
        target_calories_kcal= round(target_kcal, 0),
        recommended_menus   = result.recommended_menus,
        compliance_status   = result.compliance_status,
        global_explanation  = result.global_explanation,
        user_insights       = result.user_insights,
    )


if __name__ == "__main__":
    start_server()
