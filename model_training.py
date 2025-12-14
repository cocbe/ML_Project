import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================= CONFIG =================
TEST_SIZE = 0.20
RANDOM_SEED = 12

PCA_COMPONENTS = 500

SVM_UNKNOWN_THRESHOLD = 0.55
KNN_UNKNOWN_THRESHOLD_STD = 2  # multiplier for threshold
KNN_N_NEIGHBORS = 5
# =========================================

def train_models(X, y):
    print("\n--- Training SVM & KNN with Unknown Detection ---")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    n_components = min(PCA_COMPONENTS, X_train_scaled.shape[1], X_train_scaled.shape[0])
    if n_components < PCA_COMPONENTS:
        print(f"Adjusting PCA components from {PCA_COMPONENTS} to {n_components} (limited by data)")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA reduced features to {X_train_pca.shape[1]}")

    unknown_label = int(y.max()) + 1


    # SVM
    start = time.time()
    svm = SVC(C=10, gamma="scale", kernel="rbf", probability=True)
    svm.fit(X_train_pca, y_train)
    svm_time = time.time() - start

    svm_probs = svm.predict_proba(X_test_pca)
    svm_max_prob = np.max(svm_probs, axis=1)
    svm_preds = np.argmax(svm_probs, axis=1)
    svm_final = np.where(svm_max_prob < SVM_UNKNOWN_THRESHOLD, unknown_label, svm_preds)
    svm_acc = accuracy_score(y_test, svm_final)


    # KNN
    knn = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)
    knn.fit(X_train_pca, y_train)

    # Compute distance threshold from training data
    train_dists, _ = knn.kneighbors(X_train_pca)
    train_mean_dist = train_dists.mean(axis=1)
    knn_threshold = train_mean_dist.mean() + KNN_UNKNOWN_THRESHOLD_STD * train_mean_dist.std()

    test_dists, _ = knn.kneighbors(X_test_pca)
    test_mean_dist = test_dists.mean(axis=1)

    knn_preds = knn.predict(X_test_pca)
    knn_final = np.where(test_mean_dist > knn_threshold, unknown_label, knn_preds)
    knn_acc = accuracy_score(y_test, knn_final)
    knn_time = 0

    config = get_training_config(unknown_label)

    return {
        "X_test": X_test,
        "Y_test": y_test,
        "SVC": {
            "model": svm,
            "predictions": svm_final,
            "accuracy": svm_acc,
            "train_time_sec": round(svm_time, 2)
        },
        "KNN": {
            "model": knn,
            "predictions": knn_final,
            "accuracy": knn_acc,
            "train_time_sec": round(knn_time, 2)
        },
        "pca": pca,
        "scaler": scaler
        ,
        "training_config": config
    }


def get_training_config(unknown_label):
    config = {
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "pca_components": PCA_COMPONENTS,
        "unknown_label": unknown_label,
        "svm_unknown_threshold": SVM_UNKNOWN_THRESHOLD,
        "knn_unknown_threshold_std": KNN_UNKNOWN_THRESHOLD_STD,
        "knn_n_neighbors": KNN_N_NEIGHBORS
    }
    return config
