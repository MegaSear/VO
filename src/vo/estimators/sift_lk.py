from vo.estimators.base import BaseEstimator
import cv2
import numpy as np
class SiftLkEstimator(BaseEstimator):
    def __init__(self, nfeatures: int = 2000, ransac_threshold: float = 1.0):
        self.nfeatures = nfeatures
        self.ransac_threshold = ransac_threshold
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)

    def estimate(self, img1, img2, K):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1 = self.sift.detect(gray1, None)
   
        pts1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)
        status = status.ravel().astype(bool)

        pts1_v = pts1.reshape(-1, 2)[status]
        pts2_v = pts2.reshape(-1, 2)[status]
   

        R, t, info = self._pose_recovery(pts1_v, pts2_v, K, self.ransac_threshold)
        info.update({
            "keypoints_1": len(kp1),
            "tracked_points": np.sum(status),
            # NB: LK tracking success rate (frac. of frame-1 keypoints successfully
            # tracked into frame 2) - NOT classical detector repeatability.
            "repeatability": len(pts1_v) / len(kp1),
        })
        return R, t, info