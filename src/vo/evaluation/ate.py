import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os

def umeyama_alignment(src, dst, with_scale=True):
    # Подбирает scale через МНК
    assert src.shape == dst.shape
    n, dim = src.shape

    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    sigma_src = (src_c ** 2).sum() / n
    Sigma = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / sigma_src if with_scale else 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def compute_ate(traj_est, traj_gt, with_scale=False):
    s, R, t = umeyama_alignment(traj_est, traj_gt, with_scale=with_scale)
    traj_est_aligned = (s * (R @ traj_est.T).T) + t
    errors = np.linalg.norm(traj_est_aligned - traj_gt, axis=1)
    return {
        "ATE_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ATE_mean_m": float(np.mean(errors)),
        "ATE_median_m": float(np.median(errors)),
        "ATE_max_m": float(np.max(errors)),
        "scale_factor": float(s),
    }, traj_est_aligned


def trajectory_length(traj):
    return np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))


def run_ate_analysis(evaluator, split_name, gap=5, estimator_name=None, plot=True):
    """ ATE-анализ: строит траекторию в двух режимах (с GT-масштабом и без), считает ATE, относительный ATE. """
    traj_est_gt_scale, traj_gt, _ = evaluator.run_as_trajectory(split_name, gap=gap, use_gt_scale=True)
    ate_direction, traj_aligned_direction = compute_ate(traj_est_gt_scale, traj_gt, with_scale=False)

    traj_est_unit, traj_gt2, _ = evaluator.run_as_trajectory(split_name, gap=gap, use_gt_scale=False)
    ate_realistic, traj_aligned_realistic = compute_ate(traj_est_unit, traj_gt2, with_scale=True)

    gt_length = trajectory_length(traj_gt)
    summary = {
        "split": split_name, "gap": gap,
        "gt_trajectory_length_m": gt_length,
        "ATE_direction_only_rmse_m": ate_direction["ATE_rmse_m"],
        "ATE_direction_only_relative": ate_direction["ATE_rmse_m"] / gt_length,
        "ATE_realistic_rmse_m": ate_realistic["ATE_rmse_m"],
        "ATE_realistic_relative": ate_realistic["ATE_rmse_m"] / gt_length,
        "ATE_realistic_scale_factor": ate_realistic["scale_factor"],
    }

    print(
        f"[{split_name}, gap={gap}] GT length: {gt_length:.2f} m | "
        f"ATE direction-only: {ate_direction['ATE_rmse_m']:.3f} m ({summary['ATE_direction_only_relative']:.2%}) | "
        f"ATE realistic: {ate_realistic['ATE_rmse_m']:.3f} m ({summary['ATE_realistic_relative']:.2%}, "
        f"scale={ate_realistic['scale_factor']:.2f})"
    )

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        plt.rcParams.update({'font.size': 14})
        for ax, traj_aligned, title, ate in [
            (axes[0], traj_aligned_direction, "Direction only (GT scale)", ate_direction),
            (axes[1], traj_aligned_realistic, "Realistic (unit scale + global fit)", ate_realistic),
        ]:
            ax.plot(traj_gt[:, 0], traj_gt[:, 1], linestyle='-', label="GT")
            ax.plot(traj_aligned[:, 0], traj_aligned[:, 1], linestyle='--', label="Estimated (aligned)")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.axis("equal"); ax.legend()
            ax.set_title(f"{title}\nATE RMSE={ate['ATE_rmse_m']:.2f} m")
        plt.suptitle(f"{split_name} split, gap={gap}")
        plt.tight_layout()

        os.makedirs("figures/generated", exist_ok=True)
        fig.savefig(f"figures/generated/fig_ate_{estimator_name}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
    return summary

# =========================================================
# =================== Windowed ATE =========================
# =========================================================

def _build_window_trajectory(pose_evaluator, pairs, use_gt_scale):
    """То же накопление, что в PoseEvaluator.run_as_trajectory, но на произвольном
    списке пар (нужно, чтобы можно было сбрасывать P_est/P_gt каждые `window` шагов)."""
    P_est, P_gt = np.eye(4), np.eye(4)
    traj_est = [P_est[:3, 3].copy()]
    traj_gt = [P_gt[:3, 3].copy()]

    for idx1, idx2 in pairs:
        row = pose_evaluator._process_pair(idx1, idx2)
        t_est = row["t_est"].copy()
        if use_gt_scale:
            norm_est = np.linalg.norm(t_est)
            if norm_est < 1e-8:
                continue  # как в оригинале: пара пропускается целиком, est и gt не расходятся
            t_est = t_est * (np.linalg.norm(row["t_gt"]) / norm_est)

        T_rel_est = np.eye(4); T_rel_est[:3, :3] = row["R_est"]; T_rel_est[:3, 3] = t_est
        T_rel_gt  = np.eye(4); T_rel_gt[:3, :3]  = row["R_gt"];  T_rel_gt[:3, 3]  = row["t_gt"]

        P_est = T_rel_est @ P_est
        P_gt  = T_rel_gt  @ P_gt

        traj_est.append(np.linalg.inv(P_est)[:3, 3].copy())
        traj_gt.append(np.linalg.inv(P_gt)[:3, 3].copy())

    return np.array(traj_est), np.array(traj_gt)


def run_windowed_ate_analysis(pose_evaluator, split_name, gap=5, window=8):
    """
    Режет последовательность на непересекающиеся окна по `window` relative-pose шагов
    и в каждом окне заново стартует накопление (P_est = P_gt = I) — так один
    катастрофический шаг не размазывается по всей длине траектории, как в global ATE.
    """
    pairs = pose_evaluator.dataset.get_pair_indices_on_split(split_name, gap, gap)
    n_windows = int(np.ceil(len(pairs) / window))

    window_rows = []
    for w in range(n_windows):
        window_pairs = pairs[w * window: (w + 1) * window]
        if len(window_pairs) < 2:
            continue  # Umeyama/ATE не имеет смысла на <2 точках траектории

        traj_est_gt_scale, traj_gt = _build_window_trajectory(pose_evaluator, window_pairs, use_gt_scale=True)
        ate_direction, _ = compute_ate(traj_est_gt_scale, traj_gt, with_scale=False)

        traj_est_unit, traj_gt2 = _build_window_trajectory(pose_evaluator, window_pairs, use_gt_scale=False)
        ate_realistic, _ = compute_ate(traj_est_unit, traj_gt2, with_scale=True)

        gt_length = trajectory_length(traj_gt)
        window_rows.append({
            "window": w,
            "idx1_start": window_pairs[0][0], "idx2_end": window_pairs[-1][1],
            "n_pairs": len(window_pairs),
            "gt_length_m": gt_length,
            "ATE_direction_only_relative": ate_direction["ATE_rmse_m"] / gt_length if gt_length > 1e-8 else np.nan,
            "ATE_realistic_relative": ate_realistic["ATE_rmse_m"] / gt_length if gt_length > 1e-8 else np.nan,
            "ATE_realistic_scale_factor": ate_realistic["scale_factor"],
        })

    df = pd.DataFrame(window_rows)
    worst = df.loc[df["ATE_realistic_relative"].idxmax()]

    print(
        f"[{split_name}, gap={gap}, window={window}] n_windows={len(df)} | "
        f"median ATE_realistic_rel={df['ATE_realistic_relative'].median():.2%} | "
        f"worst ATE_realistic_rel={worst['ATE_realistic_relative']:.2%} "
        f"(window #{int(worst['window'])}, кадры {worst['idx1_start']}-{worst['idx2_end']})"
    )

    return df