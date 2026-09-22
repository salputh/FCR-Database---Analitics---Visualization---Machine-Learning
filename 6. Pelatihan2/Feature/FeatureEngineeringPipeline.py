import pandas as pd
import numpy as np
import logging
import json
from typing import List, Dict
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konfigurasi ini penting agar pipeline tahu parameter apa yang harus digunakan
FEATURE_CONFIG = {
    "lag_periods": [1, 2, 3, 7, 14],
    "rolling_windows": [3, 7, 14],
    "health_weights": {"mati": 0.5, "deplesi": 0.5},
    "decimal_places": 3,
}


class FeatureEngineeringPipeline:
    """
    Versi LENGKAP dan FINAL dari pipeline rekayasa fitur.
    Kelas ini mencerminkan 1:1 semua logika dari kelas FeatureEngineer asli.
    """

    def __init__(self, config: Dict):
        if not config:
            raise ValueError("Konfigurasi (config) harus disediakan.")
        self.config = config
        self.feature_names_ = {
            "lagged": [],
            "rolling": [],
            "interaction": [],
            "trend": [],
            "domain": [],
        }
        self.final_columns_ = []
        self.is_fitted = False

    # =============================================================================
    # METODE-METODE INTERNAL UNTUK MEMBUAT & MEMPROSES FITUR
    # (Logika sama persis dengan yang Anda berikan)
    # =============================================================================
    def _safe_numeric_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        # Kolom yang harus dikecualikan dari konversi numerik
        exclude_columns = ["TANGGAL"]

        for col in df_result.columns:
            # Skip kolom yang dikecualikan
            if col in exclude_columns:
                continue

            if df_result[col].dtype == "object":
                df_result[col] = pd.to_numeric(
                    df_result[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce",
                )
        return df_result

    def _create_lagged_features(
        self, df: pd.DataFrame, target_columns: List[str]
    ) -> pd.DataFrame:
        df_result = df.copy()
        lag_periods = self.config.get("lag_periods", [])
        for column in target_columns:
            if column in df_result.columns:
                for lag in lag_periods:
                    lag_column_name = f"{column}_lag_{lag}"
                    df_result[lag_column_name] = df_result.groupby("PERIODE")[
                        column
                    ].shift(lag)
                    self.feature_names_["lagged"].append(lag_column_name)
        return df_result

    def _create_rolling_features(
        self, df: pd.DataFrame, target_columns: List[str]
    ) -> pd.DataFrame:
        df_result = df.copy()
        windows = self.config.get("rolling_windows", [])
        for column in target_columns:
            if column in df_result.columns:
                for window in windows:
                    grouped = df_result.groupby("PERIODE")[column]
                    # Definisikan nama kolom sebelum digunakan
                    mean_col = f"{column}_rolling_mean_{window}"
                    std_col = f"{column}_rolling_std_{window}"
                    min_col = f"{column}_rolling_min_{window}"
                    max_col = f"{column}_rolling_max_{window}"

                    df_result[mean_col] = (
                        grouped.rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    df_result[std_col] = (
                        grouped.rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=0, drop=True)
                    )
                    df_result[min_col] = (
                        grouped.rolling(window=window, min_periods=1)
                        .min()
                        .reset_index(level=0, drop=True)
                    )
                    df_result[max_col] = (
                        grouped.rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
                    self.feature_names_["rolling"].extend(
                        [mean_col, std_col, min_col, max_col]
                    )
        return df_result

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        if all(c in df_result for c in ["PAKAN_ACT_GR/EK", "DG_ACT"]):
            df_result["PAKAN_DG_RATIO"] = df_result["PAKAN_ACT_GR/EK"] / (
                df_result["DG_ACT"] + 1e-8
            )
            self.feature_names_["interaction"].append("PAKAN_DG_RATIO")
        if all(c in df_result for c in ["SUHU_MAX", "KELEMBABAN"]):
            df_result["SUHU_KELEMBABAN_INTERACTION"] = (
                df_result["SUHU_MAX"] * df_result["KELEMBABAN"]
            )
            self.feature_names_["interaction"].append("SUHU_KELEMBABAN_INTERACTION")
        if all(c in df_result for c in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]):
            df_result["STRESS_INDEX"] = (
                df_result["SUHU_MAX"] - df_result["SUHU_MIN"]
            ) * df_result["KELEMBABAN"]
            self.feature_names_["interaction"].append("STRESS_INDEX")
        if "AGE" in df_result.columns:
            df_result["AGE_SQUARED"] = df_result["AGE"] ** 2
            df_result["AGE_CUBED"] = df_result["AGE"] ** 3
            self.feature_names_["interaction"].extend(["AGE_SQUARED", "AGE_CUBED"])
        if all(c in df_result for c in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]):
            temp_avg = (df_result["SUHU_MAX"] + df_result["SUHU_MIN"]) / 2
            temp_deviation = np.abs(temp_avg - 22)
            humidity_factor = np.where(
                (df_result["KELEMBABAN"] >= 60) & (df_result["KELEMBABAN"] <= 70),
                1.0,
                0.8,
            )
            df_result["COMFORT_INDEX"] = humidity_factor / (1 + temp_deviation * 0.1)
            self.feature_names_["interaction"].append("COMFORT_INDEX")
        return df_result

    def _create_trend_features(
        self, df: pd.DataFrame, target_columns: List[str]
    ) -> pd.DataFrame:
        df_result = df.copy()
        for column in target_columns:
            if column in df_result.columns and column != "FCR_ACT":
                grouped = df_result.groupby("PERIODE")[column]
                first_diff = grouped.diff(1)
                df_result[f"{column}_momentum_1"] = first_diff
                df_result[f"{column}_acceleration"] = first_diff.groupby(
                    df_result["PERIODE"]
                ).diff(1)
                df_result[f"{column}_volatility_3"] = (
                    first_diff.groupby(df_result["PERIODE"])
                    .rolling(window=3, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                df_result[f"{column}_trend_slope_3"] = (
                    grouped.rolling(window=3, min_periods=3)
                    .apply(
                        lambda x: (
                            np.polyfit(range(len(x)), x, 1)[0]
                            if len(x) == 3
                            else np.nan
                        )
                    )
                    .reset_index(level=0, drop=True)
                )
                self.feature_names_["trend"].extend(
                    [
                        f"{column}_momentum_1",
                        f"{column}_acceleration",
                        f"{column}_volatility_3",
                        f"{column}_trend_slope_3",
                    ]
                )
        return df_result

    def _create_domain_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        health_weights = self.config.get("health_weights", {})
        health_cols = list(health_weights.keys())
        if all(c in df_result for c in health_cols):
            df_result["HEALTH_STATUS_SCORE"] = sum(
                df_result[col] * weight
                for col, weight in health_weights.items()
                if col in df_result
            )
            self.feature_names_["domain"].append("HEALTH_STATUS_SCORE")
        if all(
            c in df_result.columns
            for c in ["DG_ACT_rolling_std_7", "DG_ACT_rolling_mean_7"]
        ):
            df_result["GROWTH_CONSISTENCY_INDEX"] = df_result[
                "DG_ACT_rolling_std_7"
            ] / (df_result["DG_ACT_rolling_mean_7"] + 1e-8)
            self.feature_names_["domain"].append("GROWTH_CONSISTENCY_INDEX")
        if "FCR_ACT_lag_1" in df_result.columns:
            df_result["FEED_EFFICIENCY_lag_1"] = 1.0 / (
                df_result["FCR_ACT_lag_1"] + 1e-8
            )
            self.feature_names_["domain"].append("FEED_EFFICIENCY_lag_1")
            if all(
                c in df_result.columns for c in ["SUHU_MAX", "SUHU_MIN", "KELEMBABAN"]
            ):
                temp_factor = 1 - 0.05 * (df_result["SUHU_MAX"] - df_result["SUHU_MIN"])
                humidity_factor = 1 + 0.02 * (df_result["KELEMBABAN"] - 65) / 65
                df_result["FCR_EFFICIENCY_SCORE_lag_1"] = (
                    df_result["FEED_EFFICIENCY_lag_1"] * temp_factor * humidity_factor
                )
                self.feature_names_["domain"].append("FCR_EFFICIENCY_SCORE_lag_1")
        if "AGE" in df_result.columns:
            df_result["GROWTH_STAGE"] = pd.cut(
                df_result["AGE"],
                bins=[0, 7, 14, 21, 28, 35, float("inf")],
                labels=[1, 2, 3, 4, 5, 6],
                include_lowest=True,
            ).astype(float)
            self.feature_names_["domain"].append("GROWTH_STAGE")
        return df_result

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_result = df.copy()
        for periode in df_result["PERIODE"].unique():
            mask = df_result["PERIODE"] == periode
            df_result.loc[mask] = df_result.loc[mask].ffill()
            df_result.loc[mask] = df_result.loc[mask].bfill()
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns
        columns_to_fill = [
            col for col in numeric_columns if col not in ["PERIODE", "AGE"]
        ]
        for column in columns_to_fill:
            for periode in df_result["PERIODE"].unique():
                mask = df_result["PERIODE"] == periode
                if df_result.loc[mask, column].isnull().any():
                    median_val = df_result.loc[mask, column].median()
                    if pd.isna(median_val):
                        median_val = df_result[column].median()
                    df_result.loc[mask, column] = df_result.loc[mask, column].fillna(
                        median_val
                    )
        return df_result

    def _save_feature_metadata(self, output_dir: str):
        metadata = {
            "config": self.config,
            "created_features": self.feature_names_,
        }
        Path(output_dir).mkdir(exist_ok=True)
        with open(Path(output_dir) / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(
            f"Metadata fitur disimpan di {Path(output_dir) / 'feature_metadata.json'}"
        )

    # =============================================================================
    # METODE UTAMA: FIT, TRANSFORM, DAN FIT_TRANSFORM
    # =============================================================================

    def fit(self, X: pd.DataFrame, y=None, metadata_output_dir: str = None):
        logger.info("Memulai proses 'fit' pada FeatureEngineeringPipeline...")
        self.__init__(self.config)  # Reset state setiap kali fit dijalankan

        # 1. Jalankan pipeline pembuatan fitur lengkap
        X_with_features = self._run_feature_creation_pipeline(X)

        # 2. Simpan daftar kolom final (semua kolom yang dibuat)
        self.final_columns_ = list(X_with_features.columns)

        # 3. Simpan metadata jika path disediakan
        if metadata_output_dir:
            self._save_feature_metadata(metadata_output_dir)

        self.is_fitted = True
        logger.info("Proses 'fit' selesai.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline ini belum di-'fit'. Panggil .fit() atau .fit_transform() terlebih dahulu."
            )

        logger.info(f"Memulai proses 'transform' pada data...")
        df_with_features = self._run_feature_creation_pipeline(X)

        # Pastikan kolom output konsisten dengan saat training
        for col in self.final_columns_:
            if col not in df_with_features.columns:
                df_with_features[col] = 0
        df_final = df_with_features[self.final_columns_]

        # Pembulatan data sesuai logika asli
        numeric_cols = df_final.select_dtypes(include=np.number).columns
        cols_to_round = [col for col in numeric_cols if col not in ["PERIODE", "AGE"]]
        df_final[cols_to_round] = df_final[cols_to_round].round(
            self.config.get("decimal_places", 3)
        )

        return df_final

    def fit_transform(
        self, X: pd.DataFrame, y=None, metadata_output_dir: str = None
    ) -> pd.DataFrame:
        self.fit(X, y, metadata_output_dir=metadata_output_dir)
        return self.transform(X)

    def _run_feature_creation_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper untuk menjalankan semua langkah pembuatan fitur secara berurutan."""
        df_processed = df.sort_values(["PERIODE", "AGE"]).reset_index(drop=True)
        df_processed = self._safe_numeric_conversion(df_processed)

        base_cols = [
            "PAKAN_ACT_GR/EK",
            "DG_ACT",
            "SUHU_MAX",
            "SUHU_MIN",
            "KELEMBABAN",
            "MATI",
            "JML_DEPLESI",
            "ABW_ACT",
        ]
        delta_cols = ["DELTA_FCR", "DELTA_PAKAN_GR/EKOR", "DELTA_DG"]
        lagged_columns = ["FCR_ACT"] + base_cols + delta_cols
        rolling_columns = base_cols + delta_cols
        trend_columns = base_cols + delta_cols

        df_processed = self._create_lagged_features(
            df_processed, [c for c in lagged_columns if c in df_processed.columns]
        )
        df_processed = self._create_rolling_features(
            df_processed, [c for c in rolling_columns if c in df_processed.columns]
        )
        df_processed = self._create_interaction_features(df_processed)
        df_processed = self._create_trend_features(
            df_processed, [c for c in trend_columns if c in df_processed.columns]
        )
        df_processed = self._create_domain_specific_features(df_processed)
        df_processed = self._handle_missing_values(df_processed)
        return df_processed
