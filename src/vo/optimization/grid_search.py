import itertools
from vo.evaluation.pose_eval import PoseEvaluator
def score_on(estimator_cls, params, gap, step, dataset, split_name, K):
    """Меньше — лучше. Селекционный скор по качеству позы на val."""
    estimator = estimator_cls(**params)
    pose_evaluator = PoseEvaluator(estimator, dataset, K)
    results = pose_evaluator.run_as_sequence(split_name, gap=gap, step=step)
    pose_err = results[["R_err_deg", "t_err_deg"]].max(axis=1)  # как в get_auc
    return {
        "params": params,
        "median_pose_err": pose_err.median(),
        "mean_inlier_ratio": results["inlier_ratio"].mean(),
        "n_pairs": len(results),
    }

def grid_search(estimator_cls, grid, gap, step, dataset, split_name, K):
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    scored = [score_on(estimator_cls, params, gap, step, dataset, split_name, K) for params in combos]
    scored.sort(key=lambda r: r["median_pose_err"])
    return scored