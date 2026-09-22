# FCR Database - Analitics - Visualization - Machine Learning

Sistem prediksi **FCR (Feed Conversion Ratio)** ayam broiler berbasis machine learning hybrid **TCN (Temporal Convolutional Network) + XGBoost**, lengkap dengan pipeline normalisasi data, rekayasa fitur, pelatihan model, Explainable AI (XAI), dan aplikasi web interaktif (Streamlit).

## 🐔 Tentang Proyek

FCR adalah metrik penting dalam industri peternakan ayam broiler untuk mengukur efisiensi konversi pakan menjadi bobot badan ayam. Proyek ini membangun model prediktif FCR menggunakan data historis performa harian peternakan, dengan pendekatan hybrid deep learning (TCN sebagai feature encoder) dan gradient boosting (XGBoost sebagai predictor), serta dilengkapi analisis Explainable AI (SHAP/LIME) untuk memberikan rekomendasi manajerial yang dapat ditindaklanjuti.

## ✨ Fitur Utama

- **Prediksi FCR** berbasis model hybrid TCN + XGBoost dengan berbagai horizon waktu
- **Pipeline data lengkap**: normalisasi, rekayasa fitur, dan penggabungan data multi-periode
- **Explainable AI (XAI)**: analisis SHAP/LIME untuk interpretasi model dan rekomendasi manajerial otomatis
- **Visualisasi & analitik**: evaluasi model, feature importance, dan plot analisis fitur
- **Aplikasi web interaktif** berbasis Streamlit untuk prediksi dan eksplorasi hasil
- **Hyperparameter optimization** menggunakan Optuna
- **REST API** untuk integrasi sistem (`7. Integrasi/main_api.py`)
- Modul pendukung computer vision (`camera.py`) untuk monitoring kandang berbasis webcam

## 🛠️ Tech Stack

| Kategori | Tools/Library |
|---|---|
| Deep Learning | TensorFlow, Keras-TCN |
| Machine Learning | XGBoost, scikit-learn |
| Explainable AI | SHAP, LIME |
| Hyperparameter Tuning | Optuna |
| Data Processing | Pandas, NumPy |
| Visualisasi | Matplotlib, Seaborn |
| Aplikasi Web | Streamlit |
| Computer Vision | OpenCV |

## 📁 Struktur Proyek

```
├── 1. Normalisasi/         # Pipeline normalisasi & penggabungan data mentah
├── 2. Hasil Normalisasi/   # Output data hasil normalisasi & rekayasa fitur
├── 3. non-normalize/       # Data per-periode sebelum normalisasi
├── 4. pelatihan/           # Pipeline pelatihan model & analisis XAI (v1)
├── 5. Model/               # Model terlatih (TCN encoder, XGBoost, scaler)
├── 6. Pelatihan2/          # Pipeline pelatihan model & analisis XAI (v2)
├── 7. Integrasi/           # REST API & integrasi model ke sistem produksi
├── evaluation_plots/       # Visualisasi evaluasi performa model
├── feature_analysis_plots/ # Visualisasi & laporan analisis fitur (XAI)
├── FeatureImportance/      # Hasil analisis feature importance
├── RapidMiner/             # Aset pendukung RapidMiner
├── app.py                  # Aplikasi web Streamlit untuk prediksi FCR
├── camera.py               # Modul monitoring kamera (OpenCV)
└── requirements.txt        # Daftar dependency Python
```

## 🚀 Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/salputh/FCR-Database---Analitics---Visualization---Machine-Learning.git
   cd FCR-Database---Analitics---Visualization---Machine-Learning
   ```

2. Buat virtual environment dan install dependency:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## ▶️ Menjalankan Aplikasi

Jalankan aplikasi prediksi FCR berbasis Streamlit:

```bash
streamlit run app.py
```

Untuk menjalankan REST API integrasi:

```bash
cd "7. Integrasi"
python main_api.py
```

## 🔄 Alur Pipeline

1. **Normalisasi** (`1. Normalisasi/`) — Menggabungkan dan menormalisasi data performa harian mentah dari berbagai periode pemeliharaan.
2. **Rekayasa Fitur** (`2. Hasil Normalisasi/`) — Menghasilkan fitur turunan dan metadata fitur untuk pelatihan model.
3. **Pelatihan Model** (`4. pelatihan/`, `6. Pelatihan2/`) — Melatih model hybrid TCN + XGBoost, termasuk hyperparameter tuning dan evaluasi.
4. **Explainable AI** — Menghasilkan analisis SHAP/LIME dan laporan rekomendasi manajerial berbasis prediksi model.
5. **Model Akhir** (`5. Model/`) — Menyimpan artefak model terlatih siap pakai (encoder, scaler, model hybrid).
6. **Integrasi** (`7. Integrasi/`) — Menyediakan API untuk konsumsi prediksi oleh sistem eksternal.
7. **Aplikasi** (`app.py`) — Antarmuka web untuk melakukan prediksi FCR secara interaktif.

## 📊 Data

Dataset berisi data performa harian peternakan ayam broiler (usia, jumlah ayam, konsumsi pakan, bobot, dsb.) dari berbagai periode pemeliharaan, yang telah melalui proses normalisasi dan rekayasa fitur sebelum digunakan untuk pelatihan model.

## 📄 Lisensi

Proyek ini bersifat privat untuk keperluan riset dan pengembangan internal.
