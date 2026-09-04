import cv2
import numpy as np

def compute_texture_score(img, method="laplacian_var"):
    """Дисперсия Лапласиана — прокси-метрика текстурности/резкости кадра."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if method == "laplacian_var":
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    elif method == "gradient_var":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.var(np.sqrt(gx ** 2 + gy ** 2)))
    elif method == "entropy":
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-12)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))
    else:
        raise ValueError(f"Unknown method: {method}")