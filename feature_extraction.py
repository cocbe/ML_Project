import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

# ================= CONFIG =================

# --- HOG (Shape) ---
HOG_ORIENTATIONS = 9
# revert to original pixels-per-cell
HOG_PPC = (32, 32)
HOG_CPB = (2, 2)
# --- LBP (Texture) ---
LBP_POINTS = 8
LBP_RADIUS = 1
# --- Color Histogram ---
HIST_BINS = 8            

# =========================================

def compute_color_histogram(image, mask=None, bins=HIST_BINS):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_H = cv2.calcHist([hsv], [0], mask, [bins], [0, 180])
    hist_S = cv2.calcHist([hsv], [1], mask, [bins], [0, 256])
    hist_V = cv2.calcHist([hsv], [2], mask, [bins], [0, 256])

    cv2.normalize(hist_H, hist_H)
    cv2.normalize(hist_S, hist_S)
    cv2.normalize(hist_V, hist_V)

    return np.concatenate([
        hist_H.flatten(),
        hist_S.flatten(),
        hist_V.flatten()
    ])


def extract_features(image):
    # --- Grayscale ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- HOG ---
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        transform_sqrt=True,
        visualize=False,
        feature_vector=True
    )

    # --- LBP ---
    lbp = local_binary_pattern(
        gray,
        LBP_POINTS,
        LBP_RADIUS,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2)
    )

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # --- Color Features ---
    color_features = compute_color_histogram(image)

    # --- Edge Density ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # --- Saturation  ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    sat_std = np.std(hsv[:, :, 1])

    # --- Feature Vector ---
    return np.concatenate([
        hog_features,
        lbp_hist,
        color_features,
        [edge_density, sat_mean, sat_std]
    ])
