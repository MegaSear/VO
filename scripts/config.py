from dataclasses import dataclass, field
 
import numpy as np
import torch
import os
from src.estimators.sift_essential import SiftEssentialEstimator
from src.estimators.sift_lk import SiftLkEstimator
from src.estimators.superpoint_lightglue import SuperPointLightGlueEstimator
from dotenv import load_dotenv
load_dotenv()

@dataclass
class ExperimentConfig:
    home: str = os.environ.get("VO_DATA_HOME", "./data")
    start_idx: int = 300
    end_idx: int = 500
    split_weights: dict = field(default_factory=lambda: {"train": 0.4, "val": 0.3, "test": 0.3})
 
    # Интринсики камеры ALTO
    K: np.ndarray = field(default_factory=lambda: np.array([
        [324.1, 0, 250.0],
        [0, 324.1, 250.0],
        [0, 0, 1.0],
    ], dtype=np.float64))
 
    # gap=N кадров между парой в оценке позы, step=1 -> скользящее окно
    val_split: str = "val"
    val_gap: int = 5
    val_step: int = 1
    test_split: str = "test"
    test_gap: int = 5
    test_step: int = 1
 
    # Профайлинг: своя (более разреженная) подвыборка пар — иначе инференс
    # дублируется в разы без пользы (EstimatorProfiler сам гоняет warmup+runs).
    profile_gap: int = 5
    profile_step: int = 5
    profile_n_pairs: int = 5
    profile_warmup: int = 2
    profile_runs: int = 5
 
    # Пороги failure-rate. Фиксированные значения для отчёта; альтернатива —
    # процентильный порог по val-распределению (см. percentile_threshold_from_val),
    # чтобы не подсматривать в test при выборе порога.
    rot_thresh_deg: float = 2.0
    trans_thresh_deg: float = 20.0
    diagnostic_cols: tuple = (
        "R_H_planarity", "parallax_median", "inlier_ratio",
        "F_inlier_ratio", "H_inlier_ratio", "matches", "texture_mean",
    )
 
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
 
 
GRIDS = {
    "sift_essential": {
        "cls": SiftEssentialEstimator,
        "grid": {
            "dist_ratio": [0.6, 0.7, 0.8],
            "ransac_threshold": [0.5, 0.8, 1.0, 1.5, 2.0],
            "nfeatures": [1000, 1500, 2000],
        },
    },
    "sift_lk": {
        "cls": SiftLkEstimator,
        "grid": {
            "nfeatures": [1000, 2000, 4000],
            "ransac_threshold": [0.5, 0.8, 1.0, 1.5, 2.0],
        },
    },
    "superPoint_lg": {
        "cls": SuperPointLightGlueEstimator,
        "grid": {
            "max_num_keypoints": [1024, 2048],
            "ransac_threshold": [0.5, 0.8, 1.0, 1.5, 2.0],
        },
    },
}
 
# Дефолтные параметры на случай --skip-search (например, при отладке отчётов)
FALLBACK_PARAMS = {
    "sift_essential": {"dist_ratio": 0.7, "ransac_threshold": 1.0, "nfeatures": 1500},
    "sift_lk": {"nfeatures": 2000, "ransac_threshold": 1.0},
    "superPoint_lg": {"max_num_keypoints": 2048, "ransac_threshold": 1.0},
}