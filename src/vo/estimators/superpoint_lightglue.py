import cv2
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd 
from vo.estimators.base import BaseEstimator


class SuperPointLightGlueEstimator(BaseEstimator):
    def __init__(self, max_num_keypoints: int = 2000, ransac_threshold: float = 1.0, 
                device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.ransac_threshold = ransac_threshold
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)

    @staticmethod
    def _to_tensor(img, device):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    @torch.inference_mode()
    def estimate(self, img1, img2, K):
        feats0 = self.extractor.extract(self._to_tensor(img1, self.device))
        feats1 = self.extractor.extract(self._to_tensor(img2, self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
        matches = matches01["matches"]

        kpts0, kpts1 = feats0["keypoints"], feats1["keypoints"]
        pts1 = kpts0[matches[:, 0]].cpu().numpy()
        pts2 = kpts1[matches[:, 1]].cpu().numpy()

        R, t, info = self._pose_recovery(pts1, pts2, K, self.ransac_threshold)

        info.update({
            "keypoints_1": len(kpts0),
            "keypoints_2": len(kpts1),
            "matches": len(matches),
            # NB: LightGlue match-retention rate (frac. of frame-1 keypoints that
            # received an accepted match) - NOT classical detector repeatability.
            "repeatability": float(len(matches) / max(len(kpts0), 1)),
        })
        return R, t, info
