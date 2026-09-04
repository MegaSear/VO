import numpy as np 

def compute_parallax(pts1, pts2, K, R):
    """Медианный угол параллакса (град.) между лучами в системе камеры 1."""
    Kinv = np.linalg.inv(K)
    p1h = np.hstack([pts1, np.ones((len(pts1), 1))]) @ Kinv.T
    p2h = np.hstack([pts2, np.ones((len(pts2), 1))]) @ Kinv.T
    p1n = p1h / np.linalg.norm(p1h, axis=1, keepdims=True)
    p2n = p2h / np.linalg.norm(p2h, axis=1, keepdims=True)
    p2n_in_cam1 = p2n @ R
    cos_angle = np.clip(np.sum(p1n * p2n_in_cam1, axis=1), -1, 1)
    angles = np.degrees(np.arccos(cos_angle))
    return np.median(angles)


def homography_symmetric_transfer_score(pts1, pts2, H, threshold=5.99, sigma=1.0):
    """
    ORB-SLAM-style score: симметричная ошибка репроекции через H и H^-1,
    гейтится chi2-порогом (2 dof, 95% -> 5.99). Точки вне порога вносят 0.
    """
    H_inv = np.linalg.inv(H)
    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])

    proj2 = (H @ pts1_h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    err12 = np.sum((pts2 - proj2) ** 2, axis=1) / sigma ** 2

    proj1 = (H_inv @ pts2_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    err21 = np.sum((pts1 - proj1) ** 2, axis=1) / sigma ** 2

    inliers = (err12 < threshold) & (err21 < threshold)
    score = (
        np.sum(np.where(inliers, threshold - err12, 0)) +
        np.sum(np.where(inliers, threshold - err21, 0))
    )
    return float(score), inliers


def fundamental_symmetric_transfer_score(pts1, pts2, F, threshold=3.84, sigma=1.0):
    """расстояние точка-эпиполярная линия в обе стороны (1 dof, 95% -> 3.84)."""
    N = len(pts1)
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])

    lines2 = (F @ pts1_h.T).T
    num2 = np.abs(np.sum(pts2_h * lines2, axis=1))
    den2 = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2) + 1e-12
    err12 = (num2 / den2) ** 2 / sigma ** 2

    lines1 = (F.T @ pts2_h.T).T
    num1 = np.abs(np.sum(pts1_h * lines1, axis=1))
    den1 = np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2) + 1e-12
    err21 = (num1 / den1) ** 2 / sigma ** 2

    inliers = (err12 < threshold) & (err21 < threshold)
    score = (
        np.sum(np.where(inliers, threshold - err12, 0)) +
        np.sum(np.where(inliers, threshold - err21, 0))
    )
    return float(score), inliers