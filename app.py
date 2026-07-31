"""
Streamlit Frontend — Diet DSS
=============================
Antarmuka pengguna untuk Sistem Pendukung Keputusan Rekomendasi Menu Diet.
Frontend ini TIDAK menjalankan FastAPI sama sekali — ia hanya mengirim
request HTTP ke API yang sudah berjalan terpisah (di Render).
"""

import streamlit as st
import requests

# -----------------------------------------------------------------------
# Konfigurasi URL API
# -----------------------------------------------------------------------
# Ganti dengan URL publik dari Render setelah deploy, contoh:
# API_URL = "https://rekomendasi-diet-api.onrender.com"
API_URL = "URL_DARI_RENDER"
RECOMMEND_ENDPOINT = f"{API_URL}/recommend"

st.set_page_config(page_title="Diet DSS — Rekomendasi Diet Harian", page_icon="🥗")

st.title("🥗 Diet DSS — Rekomendasi Diet Harian")
st.caption("Sistem hybrid Random Forest + Rule-Based Filtering")

# -----------------------------------------------------------------------
# Form input profil pengguna
# -----------------------------------------------------------------------
# PENTING: nama field di dalam `payload` di bawah HARUS SAMA PERSIS dengan
# field yang didefinisikan pada `UserProfile` di src/schemas.py. Contoh di
# bawah ini memakai nama field umum (age, gender, weight_kg, dst.) —
# sesuaikan dulu dengan schema Anda yang sebenarnya sebelum dipakai.
with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia", min_value=1, max_value=120, value=25)
        weight_kg = st.number_input("Berat badan (kg)", min_value=1.0, value=60.0)
        height_cm = st.number_input("Tinggi badan (cm)", min_value=1.0, value=165.0)
    with col2:
        gender = st.selectbox("Jenis kelamin", ["Male", "Female"])
        activity_level = st.selectbox(
            "Tingkat aktivitas",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"],
        )
        disease_type = st.selectbox(
            "Kondisi kesehatan",
            ["None", "Diabetes", "Hypertension", "Obesity", "Heart Disease"],
        )

    st.divider()
    target_calories = st.number_input(
        "Target kalori harian (kkal) — kosongkan 0 untuk estimasi otomatis",
        min_value=0, value=0,
    )
    top_n_menus = st.slider("Jumlah menu yang direkomendasikan", 1, 10, 5)

    submitted = st.form_submit_button("Dapatkan Rekomendasi")

# -----------------------------------------------------------------------
# Panggil API saat form disubmit
# -----------------------------------------------------------------------
if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "activity_level": activity_level,
        "disease_type": disease_type,
        "target_calories": target_calories or None,
        "top_n_menus": top_n_menus,
    }

    try:
        with st.spinner("Menghubungi API dan menyusun rekomendasi..."):
            response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            st.success("Rekomendasi berhasil dibuat!")

            st.subheader(f"Tipe Diet: {data.get('diet_type')}")
            st.write(f"Probabilitas: {data.get('diet_probability')}")
            st.write(data.get("diet_description"))
            st.metric("Target Kalori Harian", f"{data.get('target_calories_kcal')} kkal")

            st.subheader("Menu yang Direkomendasikan")
            st.write(data.get("recommended_menus"))

            with st.expander("Penjelasan & Insight"):
                st.write(data.get("global_explanation"))
                st.write(data.get("user_insights"))
        else:
            st.error(f"Gagal terhubung ke API (status {response.status_code}).")
            st.write(response.text)

    except requests.exceptions.RequestException as e:
        st.error("Tidak dapat menghubungi API. Periksa kembali API_URL di atas.")
        st.exception(e)
