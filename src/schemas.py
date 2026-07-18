"""
Input Validation — Pydantic Schemas & Validasi Profil Pengguna
==============================================================
Sesuai dengan laporan penelitian:
  - Fitur alergi TIDAK dipertimbangkan (batasan sistem)
  - Mendukung semua 20 fitur dataset diet_recommendations.csv
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal


VALID_ACTIVITY     = ["Sedentary", "Light", "Moderate", "Active", "Very_Active"]
VALID_DISEASE      = ["Diabetes", "Hypertension", "Obesity", "None"]
VALID_GENDER       = ["Male", "Female"]
VALID_SEVERITY     = ["Mild", "Moderate", "Severe"]
VALID_RESTRICTIONS = ["Low_Sugar", "Low_Sodium", "Low_Fat", "None"]
VALID_CUISINES     = ["Mexican", "Chinese", "Italian", "Indian", "Mediterranean", "None"]


class UserProfile(BaseModel):
    """Schema input profil pengguna untuk sistem DSS."""

    # ── Data Dasar
    Age          : int   = Field(..., ge=10, le=100,  description="Usia (tahun)")
    Gender       : Literal["Male", "Female"] = Field(..., description="Jenis kelamin")
    Weight_kg    : float = Field(..., gt=20, le=300,  description="Berat badan (kg)")
    Height_cm    : float = Field(..., gt=100, le=250, description="Tinggi badan (cm)")
    BMI          : Optional[float] = Field(
                       default=None, ge=10, le=70,
                       description="BMI (dihitung otomatis jika tidak diisi)")

    # Aktivitas Fisik
    Activity_Level : Literal["Sedentary","Light","Moderate","Active","Very_Active"] = Field(
                       ..., description="Tingkat aktivitas fisik harian")
    Weekly_Exercise_Hours : float = Field(
                       default=3.0, ge=0, le=40,
                       description="Jam olahraga per minggu")

    # Kondisi Medis 
    Disease_Type : Literal["Diabetes","Hypertension","Obesity","None"] = Field(
                       default="None", description="Riwayat penyakit utama")
    Severity     : Literal["Mild","Moderate","Severe"] = Field(
                       default="Mild", description="Tingkat keparahan penyakit")

    # ── Data Klinis / Laboratorium
    Glucose_mg_dL       : float = Field(
                       default=90.0, ge=40, le=600,
                       description="Kadar glukosa darah (mg/dL)")
    Blood_Pressure_mmHg : int   = Field(
                       default=120, ge=60, le=250,
                       description="Tekanan darah sistolik (mmHg)")
    Cholesterol_mg_dL   : float = Field(
                       default=180.0, ge=100, le=400,
                       description="Kadar kolesterol total (mg/dL)")

    # Data Diet & Nutrisi
    Daily_Caloric_Intake : float = Field(
                       default=2000.0, ge=500, le=6000,
                       description="Asupan kalori harian aktual (kcal)")
    Adherence_to_Diet_Plan : float = Field(
                       default=70.0, ge=0, le=100,
                       description="Tingkat kepatuhan terhadap rencana diet (%)")
    Dietary_Nutrient_Imbalance_Score : float = Field(
                       default=2.0, ge=0, le=10,
                       description="Skor ketidakseimbangan nutrisi (0=seimbang, 10=sangat tidak seimbang)")
    Dietary_Restrictions : Literal["Low_Sugar","Low_Sodium","Low_Fat","None"] = Field(
                       default="None", description="Batasan diet khusus")
    Preferred_Cuisine : Literal["Mexican","Chinese","Italian","Indian","Mediterranean","None"] = Field(
                       default="None", description="Preferensi jenis masakan")

    # ── Backward Compatibility (field lama) ────────────────────────────────────
    Blood_Sugar_mgdL       : Optional[float] = Field(
                       default=None, description="[Deprecated] Gunakan Glucose_mg_dL")
    Blood_Pressure_Systolic: Optional[int]   = Field(
                       default=None, description="[Deprecated] Gunakan Blood_Pressure_mmHg")
    Cholesterol_mgdL       : Optional[float] = Field(
                       default=None, description="[Deprecated] Gunakan Cholesterol_mg_dL")

    # Pengaturan Output
    top_n_menus     : int            = Field(default=3, ge=1, le=10,
                       description="Jumlah alternatif menu yang diminta")
    target_calories : Optional[float] = Field(
                       default=None, ge=800, le=5000,
                       description="Override kalori target (opsional)")

    @model_validator(mode="after")
    def compute_bmi_and_compat(self) -> "UserProfile":
        """Auto-hitung BMI. Handle backward compatibility untuk field lama."""
        # Auto BMI
        if self.BMI is None:
            h_m      = self.Height_cm / 100
            self.BMI = round(self.Weight_kg / (h_m ** 2), 2)

        # Backward compat: field lama → field baru
        if self.Blood_Sugar_mgdL is not None and self.Glucose_mg_dL == 90.0:
            self.Glucose_mg_dL = self.Blood_Sugar_mgdL
        if self.Blood_Pressure_Systolic is not None and self.Blood_Pressure_mmHg == 120:
            self.Blood_Pressure_mmHg = self.Blood_Pressure_Systolic
        if self.Cholesterol_mgdL is not None and self.Cholesterol_mg_dL == 180.0:
            self.Cholesterol_mg_dL = self.Cholesterol_mgdL

        return self

    def to_dict(self) -> dict:
        return self.model_dump()


class DSSResponse(BaseModel):
    """Schema output lengkap sistem DSS."""
    status               : str
    diet_type            : str
    diet_probability     : dict
    diet_description     : str
    target_calories_kcal : float
    recommended_menus    : list
    compliance_status    : str
    global_explanation   : list
    user_insights        : dict 
