from estimators.base import BaseEstimator
import cv2
import numpy as np

class SiftEssentialEstimator(BaseEstimator):
    def __init__(self, dist_ratio=0.6, ransac_threshold=1.0, nfeatures=2000):
        self.dist_ratio = dist_ratio
        self.ransac_threshold = ransac_threshold
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher()
        
    def estimate(self, img1, img2, K):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        matches = self.matcher.knnMatch(des1, des2, k=2)

        good = [
            m for m,n in matches
            if m.distance < self.dist_ratio*n.distance
        ]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        R,t,info = self._pose_recovery(pts1, pts2, K, self.ransac_threshold)

        info.update({
            "keypoints_1":len(kp1),
            "keypoints_2":len(kp2),
            "matches": len(matches),
            # NB: ratio-test survival rate (frac. of frame-1 keypoints whose best SIFT
            # match passed Lowe's ratio test) - NOT classical detector repeatability.
            # See the notes cell at the top of the notebook.
            "repeatability":len(good)/len(kp1)
        })
        return R,t,info
