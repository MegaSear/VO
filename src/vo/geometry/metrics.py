import numpy as np

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.0+ / <2.0 совместимость

# Ошибка вращения
def rotation_error_deg(R_est, R_gt):
    R_error = R_est @ R_gt.T
    trace = np.trace(R_error)
    cos_angle = np.clip((trace - 1) / 2,-1.0,1.0)
    return np.degrees(np.arccos(cos_angle))

# Ошибка направления трансляции
def translation_direction_error_deg(t_est, t_gt):
    t_est = np.asarray(t_est, dtype=float).reshape(-1)
    t_gt = np.asarray(t_gt, dtype=float).reshape(-1)
    t_est_n = t_est / np.linalg.norm(t_est)
    t_gt_n = t_gt / np.linalg.norm(t_gt)
    cos_angle = np.clip(np.dot(t_est_n, t_gt_n), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# Lindenberger, Sarlin, Pollefeys.** *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023.
# Wang, Q.** *Understanding and Optimizing Attention-Based Sparse Matching for Diverse Local Features*, 2026.
# Обе статьи используют один и тот же протокол Relative Pose Estimation AUC на MegaDepth-1500: 
# для каждой пары кадров считается максимальная угловая ошибка (между вращением и направлением трансляции), 
# затем — площадь под CDF этой ошибки, нормированная порогом K (K = 5°, 10°, 20°). 
# Оценка позы делается через Essential Matrix + LO-RANSAC — то есть тот же геометрический аппарат, 
# что и в `sift_essential`/`superPoint_lg` здесь.

def auc_at_k(errors, k):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return np.nan
    # F(theta) = fraction of errors <= theta
    thresholds = np.sort(np.unique(np.concatenate([[0.0], errors[errors <= k], [k]])))
    F = np.array([np.mean(errors <= theta) for theta in thresholds])
    return float(_TRAPZ(F, thresholds) / k) # AUC normalized by K