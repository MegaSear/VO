"""
Основной эксперимент: сравнение трёх монокулярных VO-эстиматоров
(SIFT+Essential, SIFT+LK, SuperPoint+LightGlue) на датасете ALTO.

Этапы:
  1. Hyperparameter grid search на val-сплите (для каждого эстиматора отдельно)
  2. Оценка точности позы на test-сплите (holdout, не участвовал в подборе)
  3. ATE / windowed ATE анализ
  4. Профайлинг latency / CPU & GPU памяти на небольшой фиксированной выборке пар
  5. Итоговая сводная таблица + failure-rate анализ

Запуск:
    python run_experiment.py
    python run_experiment.py --skip-search --skip-profile   # для быстрой отладки отчётов
"""
import argparse
import logging

import numpy as np
import pandas as pd

from config import ExperimentConfig, GRIDS, FALLBACK_PARAMS
from src.datasets.alto import AltoDataset
from src.estimators.sift_essential import SiftEssentialEstimator
from src.estimators.sift_lk import SiftLkEstimator
from src.estimators.superpoint_lightglue import SuperPointLightGlueEstimator
from src.evaluation.pose_eval import PoseEvaluator
from src.evaluation.failure_analysis import failure_rate_report
from src.optimization.grid_search import grid_search
from src.evaluation.statistics import get_stat, get_auc, get_rpe, format_auc, format_rpe
from src.evaluation.ate import run_ate_analysis, run_windowed_ate_analysis
from src.profiling.profiler import EstimatorProfiler

log = logging.getLogger("run_experiment")


# =========================================================
# ======================= Pipeline steps ====================
# =========================================================
def load_dataset(cfg: ExperimentConfig) -> AltoDataset:
    df = pd.read_csv(f"{cfg.home}/query.csv")
    df["idx"] = df["name"].str.extract(r"(\d+)").astype(int)
    df = df[cfg.start_idx:cfg.end_idx].reset_index(drop=True)
    df["idx"] = df["idx"] - cfg.start_idx
    return AltoDataset(df, f"{cfg.home}/frames/", splits_weights=cfg.split_weights)


def run_val_search(dataset: AltoDataset, cfg: ExperimentConfig) -> tuple[dict, dict]:
    best_params, val_search_logs = {}, {}
    for name, spec in GRIDS.items():
        log.info("val hyperparameter search: %s", name)
        scored = grid_search(spec["cls"], spec["grid"], cfg.val_gap, cfg.val_step, dataset, cfg.val_split, cfg.K)
        val_search_logs[name] = pd.DataFrame(scored)
        best_params[name] = scored[0]["params"]
        log.info("%s\nBEST: %s", val_search_logs[name].to_string(), best_params[name])
    return best_params, val_search_logs


def build_estimators(best_params: dict, device: str) -> dict:
    return {
        "sift_essential": SiftEssentialEstimator(**best_params["sift_essential"]),
        "sift_lk": SiftLkEstimator(**best_params["sift_lk"]),
        "superpoint_lg": SuperPointLightGlueEstimator(**best_params["superPoint_lg"], device=device),
    }


def run_test_evaluation(estimators: dict, dataset: AltoDataset, cfg: ExperimentConfig):
    pose_evaluators, results_by_name, ate_by_name, windowed_ate_by_name = {}, {}, {}, {}
    for name, estimator in estimators.items():
        log.info("evaluating: %s", name)
        pose_evaluator = PoseEvaluator(estimator, dataset, cfg.K)
        results = pose_evaluator.run_as_sequence(cfg.test_split, gap=cfg.test_gap, step=cfg.test_step)
        pose_evaluators[name] = pose_evaluator
        results_by_name[name] = results
        log.info("\n%s", get_stat(results).to_string())
        log.info("%s", format_auc(get_auc(results)))
        log.info("%s", format_rpe(get_rpe(results)))
        ate_by_name[name] = run_ate_analysis(pose_evaluator, cfg.test_split, gap=cfg.test_gap, estimator_name=name, plot=True)
        windowed_ate_by_name[name] = run_windowed_ate_analysis(
            pose_evaluator, cfg.test_split, gap=cfg.test_gap, window=3
        )
    return pose_evaluators, results_by_name, ate_by_name, windowed_ate_by_name


def profile_estimators(estimators: dict, dataset: AltoDataset, cfg: ExperimentConfig) -> dict:
    profile_pairs = dataset.get_pair_indices_on_split(
        cfg.test_split, gap=cfg.profile_gap, step=cfg.profile_step
    )[:cfg.profile_n_pairs]

    profile_summary = {}
    for name, estimator in estimators.items():
        log.info("profiling: %s", name)
        # NB: runs=%d — небольшая выборка, ожидается несколько % джиттера
        # между запусками на CPU. Цифры в README/research_note — из
        # конкретного прогона, не lab-grade бенчмарк.
        stats = EstimatorProfiler.profile_on_sample(
            estimator, dataset, cfg.K, profile_pairs,
            warmup=cfg.profile_warmup, runs=cfg.profile_runs,
        )
        profile_summary[name] = pd.DataFrame(stats)
        log.info("%s", profile_summary[name].agg(["mean", "median"]).to_string())
    return profile_summary


def build_overview(estimators: dict, results_by_name: dict, profile_summary: dict,
                    ate_by_name: dict, windowed_ate_by_name: dict) -> pd.DataFrame:
    rows = []
    for name in estimators:
        r, p = results_by_name[name], profile_summary[name]
        rows.append({
            "estimator": name,
            "R_err_deg_median": r["R_err_deg"].median(),
            "t_err_deg_median": r["t_err_deg"].median(),
            "inlier_ratio_mean": r["inlier_ratio"].mean(),
            "repeatability_mean": r.get("repeatability", pd.Series(dtype=float)).mean(),
            "ATE_realistic_rel": ate_by_name[name]["ATE_realistic_relative"],
            "ATE_realistic_rel_windowed": windowed_ate_by_name[name]["ATE_realistic_relative"].median(),
            "latency_ms_median": p["median_time_ms"].median(),
            "cpu_mem_mb_median": p["median_cpu_mem_mb"].median(),
            "gpu_mem_mb_median": p["median_gpu_mem_mb"].median(),
        })
    return pd.DataFrame(rows).set_index("estimator")


def run_failure_analysis(results_by_name: dict, cfg: ExperimentConfig) -> pd.DataFrame:
    failure_dfs = {
        name: failure_rate_report(results, name, cfg.rot_thresh_deg, cfg.trans_thresh_deg,
                                   list(cfg.diagnostic_cols))
        for name, results in results_by_name.items()
    }
    return pd.DataFrame({
        name: {
            "failure_rate": df["is_failure"].mean(),
            "n_failures": int(df["is_failure"].sum()),
            "n_pairs": len(df),
        }
        for name, df in failure_dfs.items()
    }).T


# =========================================================
# ========================== main ===========================
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-search", action="store_true",
                        help="Пропустить grid search на val, использовать FALLBACK_PARAMS")
    parser.add_argument("--skip-profile", action="store_true",
                        help="Пропустить профайлинг latency/памяти")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = ExperimentConfig()
    dataset = load_dataset(cfg)

    if args.skip_search:
        log.info("skipping val search, using FALLBACK_PARAMS")
        best_params = FALLBACK_PARAMS
    else:
        best_params, _val_search_logs = run_val_search(dataset, cfg)

    estimators = build_estimators(best_params, cfg.device)

    _pose_evaluators, results_by_name, ate_by_name, windowed_ate_by_name = run_test_evaluation(
        estimators, dataset, cfg
    )

    if args.skip_profile:
        log.info("skipping profiling")
        profile_summary = {
            name: pd.DataFrame([{
                "median_time_ms": np.nan, "median_cpu_mem_mb": np.nan, "median_gpu_mem_mb": np.nan,
            }])
            for name in estimators
        }
    else:
        profile_summary = profile_estimators(estimators, dataset, cfg)

    overview = build_overview(estimators, results_by_name, profile_summary, ate_by_name, windowed_ate_by_name)
    log.info("\n%s", overview.to_string())

    failure_summary = run_failure_analysis(results_by_name, cfg)
    log.info("\n%s SUMMARY %s\n%s", "=" * 15, "=" * 15, failure_summary.to_string())


if __name__ == "__main__":
    main()