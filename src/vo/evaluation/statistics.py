import numpy as np
import pandas as pd

from vo.geometry.metrics import auc_at_k

DEFAULT_STAT_METRICS = (
    "R_err_deg", "t_err_deg",
    "matches", "inliers", "inlier_ratio",
    "F_inliers", "F_inlier_ratio",
    "H_inliers", "H_inlier_ratio", "R_H_planarity",
    "repeatability", "parallax_median", "texture_mean",
)
DEFAULT_STAT_FILTERS = ("mean", "median", "min", "max")


def get_stat(results, metrics=DEFAULT_STAT_METRICS, filters=DEFAULT_STAT_FILTERS) -> pd.DataFrame:
    """Сводная статистика по колонкам, которые реально есть в results.
    Набор метрик отличается между эстиматорами (например, SIFT+LK не считает
    keypoints_2 — вместо этого у него tracked_points), поэтому берём пересечение.

    NB: "repeatability" here is per-estimator correspondence/match retention rate
    (ratio-test survival / LK-tracking success / LightGlue match acceptance) - not
    classical multi-view detector repeatability. See notes cell at notebook top.

    Возвращает DataFrame (metrics x filters), а не строку — форматирование
    для вывода/лога делает вызывающий код.
    """
    available = [m for m in metrics if m in results.columns]
    return results[available].agg(list(filters))


def get_rpe(results) -> dict:
    """Relative Pose Error: RMSE/MAE/MdAE углового отклонения R и направления t
    по всем парам последовательности (см. R_err_deg/t_err_deg из PoseEvaluator._process_pair).

    Возвращает dict с метриками вместо печати — вызывающий код решает,
    логировать их, положить в DataFrame или сравнить между эстиматорами.
    """
    rot = results["R_err_deg"].dropna().values
    trans = results["t_err_deg"].dropna().values

    return {
        "rpe_rot_rmse": float(np.sqrt(np.mean(rot ** 2))),
        "rpe_rot_mae": float(np.mean(np.abs(rot))),
        "rpe_rot_mdae": float(np.median(np.abs(rot))),
        "rpe_trans_dir_rmse": float(np.sqrt(np.mean(trans ** 2))),
        "rpe_trans_dir_mae": float(np.mean(np.abs(trans))),
        "rpe_trans_dir_mdae": float(np.median(np.abs(trans))),
    }


def get_auc(results, ks=(5, 10, 20)) -> dict:
    """AUC@K по протоколу relative pose estimation с MegaDepth-1500
    (см. комментарий к auc_at_k выше — тот же геометрический аппарат: E-matrix + LO-RANSAC).

    Возвращает {k: {"auc": ..., "accuracy": ...}} вместо печати.
    """
    pose_error = results[["R_err_deg", "t_err_deg"]].max(axis=1)
    return {
        k: {
            "auc": float(auc_at_k(pose_error, k)),
            "accuracy": float(np.mean(pose_error <= k)),
        }
        for k in ks
    }


def format_rpe(rpe: dict) -> str:
    """Строковое представление результата get_rpe — для print/log."""
    return (
        f"RPE_rot_rmse={rpe['rpe_rot_rmse']:.3f} | "
        f"RPE_rot_mae={rpe['rpe_rot_mae']:.3f} | "
        f"RPE_rot_mdae={rpe['rpe_rot_mdae']:.3f} | "
        f"RPE_trans_dir_rmse={rpe['rpe_trans_dir_rmse']:.3f} | "
        f"RPE_trans_dir_mae={rpe['rpe_trans_dir_mae']:.3f} | "
        f"RPE_trans_dir_mdae={rpe['rpe_trans_dir_mdae']:.3f}"
    )


def format_auc(auc_by_k: dict) -> str:
    """Строковое представление результата get_auc — для print/log."""
    return "\n".join(
        f"K={k:3d}° | AUC={v['auc']:.4f} | Accuracy={v['accuracy']:.4f}"
        for k, v in auc_by_k.items()
    )