import numpy as np
from tqdm import tqdm
import pandas as pd
from vo.geometry.metrics import rotation_error_deg, translation_direction_error_deg
from vo.geometry.texture import compute_texture_score

class PoseEvaluator():
    def __init__(self, estimator, dataset, K):
        self.estimator = estimator
        self.dataset = dataset 
        self.K = K

    def _process_pair(self, idx1, idx2):
        (img1, _, _), (img2, _, _) = self.dataset[idx1], self.dataset[idx2]
        R_gt, t_gt = self.dataset.gt_relative(idx1, idx2)
        R_est, t_est, info = self.estimator.estimate(img1, img2, self.K)
        t_est = np.asarray(t_est, dtype=float).reshape(3)

        return {"idx1": idx1, "idx2": idx2, **info,
            "R_est": R_est, "t_est": t_est, "R_gt": R_gt, "t_gt": t_gt,
            "R_err_deg": rotation_error_deg(R_est, R_gt),
            "t_err_deg": translation_direction_error_deg(t_est, t_gt),
            "texture_mean": (compute_texture_score(img1) + compute_texture_score(img2)) / 2.0,
        }
    
    def run_as_sequence(self, split_name, gap, step):
        pairs = self.dataset.get_pair_indices_on_split(split_name, gap, step)
        rows = [self._process_pair(i1, i2) for i1, i2 in tqdm(pairs, desc=f"Processing [{split_name}]")]
        df = pd.DataFrame(rows)
        return df
    
    def run_as_trajectory(self, split_name, gap, use_gt_scale=True):
        pairs = self.dataset.get_pair_indices_on_split(split_name, gap, gap)
        P_est, P_gt = np.eye(4), np.eye(4)
        traj_est = [P_est[:3, 3].copy()]
        traj_gt = [P_gt[:3, 3].copy()]
        step_rows = []

        for idx1, idx2 in tqdm(pairs, desc=f"Processing [{split_name}]"):
            row = self._process_pair(idx1, idx2)

            t_est = row["t_est"].copy()
            if use_gt_scale:
                norm_est = np.linalg.norm(t_est)
                if norm_est < 1e-8:
                    step_rows.append(row)
                    continue
                t_est = t_est * (np.linalg.norm(row["t_gt"]) / norm_est)

            T_rel_est = np.eye(4)
            T_rel_est[:3, :3] = row["R_est"]
            T_rel_est[:3, 3] = t_est
        
            T_rel_gt = np.eye(4)
            T_rel_gt[:3, :3] = row["R_gt"]
            T_rel_gt[:3, 3] = row["t_gt"]

            P_est = T_rel_est @ P_est
            P_gt = T_rel_gt @ P_gt

            traj_est.append(np.linalg.inv(P_est)[:3, 3].copy())
            traj_gt.append(np.linalg.inv(P_gt)[:3, 3].copy())
            step_rows.append(row)

        return np.array(traj_est), np.array(traj_gt), pd.DataFrame(step_rows)