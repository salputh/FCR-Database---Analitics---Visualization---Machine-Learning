# main_api.py

import pandas as pd
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Impor kelas HybridPredictor Anda dari skrip prediksi
# Pastikan file PredictTestFeatureEvaluation.py berada di direktori yang sama
# atau bisa diakses oleh Python path.
try:
    from PredictTestFeatureEvaluation import HybridPredictor
except ImportError as e:
    print(
        f"Error: Gagal mengimpor HybridPredictor. Pastikan file berada di path yang benar. {e}"
    )
    exit()

# =============================================================================
# SETUP APLIKASI DAN MODEL
# =============================================================================

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="API Prediksi FCR Ayam Broiler",
    description="API untuk mengintegrasikan model TCN-XGBoost dengan sistem Laravel.",
    version="1.0.0",
)

# Definisikan path ke model dan file konfigurasi Anda
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "5. Model", "models_terbaru")
CONFIG_DIR = os.path.join(BASE_DIR, "2. Hasil Normalisasi", "Terbaru")

# Muat model saat aplikasi dimulai.
# Ini memastikan model hanya dimuat sekali, bukan setiap kali ada request.
try:
    predictor = HybridPredictor(models_dir=MODELS_DIR, config_dir=CONFIG_DIR)
    print(">>> Model HybridPredictor berhasil dimuat dan siap menerima permintaan. <<<")
except Exception as e:
    print(f"FATAL: Gagal memuat model saat startup: {e}")
    predictor = None  # Set ke None jika gagal

# =============================================================================
# DEFINISI STRUKTUR DATA (Pydantic Models)
# =============================================================================


# Mendefinisikan struktur data untuk satu baris data historis/baru
# Sesuaikan nama field ini dengan nama kolom di database Laravel Anda.
class DataRow(BaseModel):
    PERIODE: int
    AGE: int
    FCR_ACT: float
    PAKAN_ACT_GR_EK: float = Field(
        alias="PAKAN_ACT_GR/EK"
    )  # alias untuk field dengan '/'
    DG_ACT: float
    SUHU_MAX: float
    SUHU_MIN: float
    KELEMBABAN: float
    MATI: int
    JML_DEPLESI: int
    ABW_ACT: float
    DELTA_FCR: float
    DELTA_PAKAN_GR_EKOR: float = Field(alias="DELTA_PAKAN_GR/EKOR")
    DELTA_DG: float

    # Memungkinkan alias digunakan
    class Config:
        allow_population_by_field_name = True


# Mendefinisikan struktur input untuk endpoint /predict
class PredictionInput(BaseModel):
    historical_data: List[DataRow]
    new_day_data: List[
        DataRow
    ]  # Meskipun hanya 1 hari, kita gunakan list untuk konsistensi
    horizon: int = 1  # Horizon default adalah 1 hari, bisa diubah


# =============================================================================
# ENDPOINT API
# =============================================================================


@app.get("/")
def read_root():
    """Endpoint utama untuk mengecek apakah API berjalan."""
    return {"status": "success", "message": "Selamat Datang di API Prediksi FCR!"}


@app.post("/predict")
async def get_prediction(data: PredictionInput) -> Dict[str, Any]:
    """
    Endpoint utama untuk melakukan prediksi FCR.
    Menerima data historis dan data hari ini, lalu mengembalikan prediksi.
    """
    if not predictor:
        raise HTTPException(
            status_code=503, detail="Model tidak tersedia atau gagal dimuat."
        )

    try:
        # 1. Konversi data input Pydantic menjadi DataFrame Pandas
        # FastAPI secara otomatis akan memvalidasi data yang masuk sesuai struktur DataRow
        df_historical = pd.DataFrame(
            [row.dict(by_alias=True) for row in data.historical_data]
        )
        df_new_day = pd.DataFrame(
            [row.dict(by_alias=True) for row in data.new_day_data]
        )

        # Pastikan tidak ada data kosong yang dikirim
        if df_historical.empty or df_new_day.empty:
            raise HTTPException(
                status_code=400,
                detail="Data historis dan data hari ini tidak boleh kosong.",
            )

        # 2. Panggil metode prediksi dari kelas HybridPredictor Anda
        # Kita gunakan `predict_single_sample` karena lebih efisien (tanpa visualisasi)
        predicted_fcr = predictor.predict_single_sample(
            new_day_data=df_new_day, historical_data=df_historical, horizon=data.horizon
        )

        # 3. Kembalikan hasil dalam format JSON
        return {
            "status": "success",
            "prediction_horizon": data.horizon,
            "predicted_fcr": predicted_fcr,
            "input_summary": {
                "historical_days": len(df_historical),
                "new_day_periode": df_new_day["PERIODE"].iloc[0],
                "new_day_age": df_new_day["AGE"].iloc[0],
            },
        }

    except ValueError as ve:
        # Menangkap error spesifik dari pipeline Anda, misal data tidak cukup
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Menangkap error lainnya
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal saat prediksi: {str(e)}",
        )


# =============================================================================
# MENJALANKAN SERVER API
# =============================================================================
if __name__ == "__main__":
    # Perintah untuk menjalankan: uvicorn main_api:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)
