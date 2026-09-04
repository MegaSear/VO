import cv2 
import numpy as np
from scipy.spatial.transform import Rotation

class AltoDataset():
    def __init__(self, df, home, splits_weights={"train": 0.6, "val": 0.2, "test": 0.2}):
        self.df = df
        self.home = home
        
        # Contiguous on time
        idx_min, idx_max = int(self.df["idx"].min()), int(self.df["idx"].max())
        span = idx_max - idx_min  
        train_end = idx_min + int(span * splits_weights["train"])
        val_end = idx_min + int(span * (splits_weights["train"] + splits_weights["val"]))

        self.splits = {
            "train": range(idx_min, train_end),
            "val": range(train_end, val_end),
            "test": range(val_end, idx_max + 1),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        R, p = self.get_pose(idx)
        return img, R, p

    def get_image(self, idx):
        row = self.df.iloc[idx]
        return cv2.imread(self.home+f"{row['name']}")
    
    def get_pose(self, idx):
        row = self.df.iloc[idx]
        R = self.quat_to_R(row)
        p = np.array([
            row["easting"],
            row["northing"],
            row["altitude"],
        ])
        return R, p
    
    # Quaternion -> rotation matrix
    def quat_to_R(self, row):
        q = [
            row["orient_x"],
            row["orient_y"],
            row["orient_z"],
            row["orient_w"],
        ]

        return Rotation.from_quat(q).as_matrix()

    # GT относительное движение
    def gt_relative(self, idx1, idx2):
        R_w_1, p1 = self.get_pose(idx1)
        R_w_2, p2 = self.get_pose(idx2)
        R_gt = R_w_2.T @ R_w_1
        t_gt = R_w_2.T @ (p1 - p2)
        return R_gt, t_gt

    def get_pair_indices_on_split(self, split_name, gap, step):
        indices = list(self.splits[split_name])
        pairs = [
            (indices[i], indices[i + gap])
            for i in range(0, len(indices) - gap, step)
        ]
        return pairs