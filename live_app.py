import os
import cv2
import json
import numpy as np
import joblib
from feature_extraction import extract_features

# ================= CONFIG =================

MODELS_FOLDER = "models"
CAMERA_FOLDER = "camera"

MODEL_TYPE = "SVC"      # "SVC" or "KNN"

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CLIP_ZSCORE = 5.0

TIME_BETWEEN_IMAGES = 1000 # in ms
# =========================================


def load_stats():
    path = os.path.join(MODELS_FOLDER, "run_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError("run_stats.json not found")
    with open(path, "r") as f:
        return json.load(f)


def print_summary(stats):
    print("\n=== Live App Info ===")
    print("Classes:", ", ".join(stats["class_names"]))
    print("Image size:", stats["image_size"])
    print("Model type:", MODEL_TYPE)
    print("Unknown label:", stats["training_config"]["unknown_label"])


def preprocess(image, image_size, scaler, pca):
    if image.shape[:2] != tuple(image_size):
        image = cv2.resize(image, tuple(image_size))

    features = extract_features(image).reshape(1, -1)
    features = scaler.transform(features)
    features = np.clip(features, -CLIP_ZSCORE, CLIP_ZSCORE)
    return pca.transform(features)


def predict(image, stats, model, scaler, pca):
    cfg = stats["training_config"]
    unknown_label = cfg["unknown_label"]

    features_pca = preprocess(
        image,
        stats["image_size"],
        scaler,
        pca
    )

    # SVC
    if MODEL_TYPE == "SVC":
        probs = model.predict_proba(features_pca)
        pred = np.argmax(probs)
        if np.max(probs) < cfg["svm_unknown_threshold"]:
            pred = unknown_label

    # KNN
    else:
        dists, _ = model.kneighbors(features_pca)
        mean_dist = dists.mean()
        threshold = cfg["knn_unknown_threshold_std"]
        pred = model.predict(features_pca)[0]
        if mean_dist > threshold:
            pred = unknown_label

    return stats["class_names"][pred]


def display(image, label):
    h, w = image.shape[:2]
    scale = min(DISPLAY_WIDTH / w, DISPLAY_HEIGHT / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    cv2.putText(
        resized,
        f"Prediction: {label}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2
    )

    cv2.imshow("Live Prediction", resized)


def run_camera(stats, model, scaler, pca):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = predict(frame, stats, model, scaler, pca)
        display(frame, label)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_folder(stats, model, scaler, pca):
    files = sorted(
        f for f in os.listdir(CAMERA_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )

    if not files:
        print("No images found")
        return

    for f in files:
        img = cv2.imread(os.path.join(CAMERA_FOLDER, f))
        if img is None:
            continue

        label = predict(img, stats, model, scaler, pca)
        display(img, label)

        if cv2.waitKey(TIME_BETWEEN_IMAGES) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    stats = load_stats()
    print_summary(stats)

    svm = joblib.load(os.path.join(MODELS_FOLDER, "svm_model.pkl"))
    knn = joblib.load(os.path.join(MODELS_FOLDER, "knn_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_FOLDER, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODELS_FOLDER, "pca.pkl"))

    model = svm if MODEL_TYPE == "SVC" else knn

    print("\n1) Camera\n2) Folder")
    choice = input("Select: ").strip()

    if choice == "1":
        run_camera(stats, model, scaler, pca)
    elif choice == "2":
        run_folder(stats, model, scaler, pca)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
