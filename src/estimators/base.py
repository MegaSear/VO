from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional

import cv2
import numpy as np
from geometry.epipolar import (
    homography_symmetric_transfer_score,
    fundamental_symmetric_transfer_score,
    compute_parallax
)

class IPoseEstimator(ABC):
    @abstractmethod
    def estimate(self, img1: np.ndarray, img2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        raise NotImplementedError
    
class BaseEstimator(IPoseEstimator):
    @classmethod
    def _pose_recovery(cls, pts1, pts2, K, ransac_threshold=1.0):
        
        # Fundamental matrix
        F, mask_F = cls._compute_fundamentalM(
            pts1,
            pts2,
            threshold=ransac_threshold
        )
        mask_F = mask_F.ravel().astype(bool)
        F_inliers = int(mask_F.sum())
        F_inlier_ratio = F_inliers / len(pts1)

        # Essential matrix
        E, mask_E = cls._compute_essentialM(pts1, pts2, cameraMatrix=K, threshold=ransac_threshold)

        # Pose recovery
        num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask_E.copy())

        # Homography
        try:
            H, mask_H = cls._compute_homography(pts1, pts2, threshold=ransac_threshold)
            H_inliers = int(mask_H.ravel().astype(bool).sum())
            H_inlier_ratio = H_inliers / len(pts1)

            score_H, _ = homography_symmetric_transfer_score(pts1, pts2, H)
            score_F, _ = fundamental_symmetric_transfer_score(pts1, pts2, F)
            # ORB-SLAM-style model-selection score (Mur-Artal et al., 2015): ratio of
            # homography- vs. fundamental-matrix symmetric transfer scores, used there
            # to detect near-planar/near-degenerate scenes during monocular init
            # (threshold ~0.45). Not a standalone geometric planarity measure.
            R_H = score_H / (score_H + score_F + 1e-12)
        except RuntimeError:
            H_inliers, H_inlier_ratio, R_H = 0, 0.0, np.nan
            
        # Parallax
        if F_inliers >= 3:
            parallax_median = compute_parallax(pts1[mask_F],pts2[mask_F],K,R)
        else:
            parallax_median = np.nan

        info = {
            "matches": len(pts1),
            "inliers": int(num_inliers),
            "inlier_ratio": float(num_inliers / len(pts1)),
            "F_inliers": F_inliers,
            "F_inlier_ratio": float(F_inlier_ratio),
            "H_inliers": H_inliers,
            "H_inlier_ratio": float(H_inlier_ratio),
            "R_H_planarity": float(R_H),
            "parallax_median": float(parallax_median)
        }
        return R, t, info
    
    @staticmethod
    def _compute_essentialM(pts1, pts2, cameraMatrix, threshold, method=cv2.RANSAC):
        E, mask_E = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=cameraMatrix,
            method=method,
            prob=0.999,
            threshold=threshold
        )
        if E is None or mask_E is None:
            raise RuntimeError("Failed to estimate Essential Matrix")
        return E, mask_E
    
    @staticmethod
    def _compute_homography(pts1, pts2, threshold, method=cv2.RANSAC):
        H, mask_H = cv2.findHomography(
            pts1, pts2,
            method=method,
            ransacReprojThreshold=threshold,
            confidence=0.999
        )
        if H is None or mask_H is None:
            raise RuntimeError("Failed to estimate Homography")
        return H, mask_H
    
    @staticmethod
    def _compute_fundamentalM(pts1, pts2, threshold, method=cv2.RANSAC):
        F, mask_F = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=method,
            ransacReprojThreshold=threshold,
            confidence=0.999
        )
        if F is None or mask_F is None:
            raise RuntimeError("Failed to estimate Fundamental Matrix")
        return F, mask_F
