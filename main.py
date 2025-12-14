import os
import time
import json
import joblib
from count import count_images
from augment import load_augmented_dataset,balance_dataset
from data_preparation import prepare_data
from model_training import train_models
from sklearn.metrics import classification_report

# ================= CONFIG =================

DATASET_PATH = "dataset"
AUGMENTED_PATH = "dataset_aug"
DEST_FOLDER = "saved_runs"
TARGET_COUNT = 1250
IMAGE_RESIZE = (256, 256)

# =========================================


def run_full_pipeline():
    print("\n===== STARTING PIPELINE =====")

    if not os.path.exists(DATASET_PATH):
        print("Dataset not found.")
        return

    # Count original images and load them in RAM
    counts = count_images(DATASET_PATH, IMAGE_RESIZE)
    print("Original counts:", counts)

    # Load augmented dataset or create them

    # all_pixels = load_augmented_dataset(AUGMENTED_PATH, resize=IMAGE_RESIZE) # this function is used to load the already created augmented images
    all_pixels = balance_dataset(DATASET_PATH,counts,AUGMENTED_PATH,TARGET_COUNT,True,IMAGE_RESIZE)  # this function is used to create new augmented images
    
    X, y, label_map = prepare_data(
        all_pixels,
        IMAGE_SIZE=IMAGE_RESIZE
    )

    # Build ordered class list
    ordered_class_names = [
        name for name, _ in sorted(label_map.items(), key=lambda x: x[1])
    ]
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")



    # Train models ( SVC + KNN )
    results = train_models(X, y)
    X_test = results["X_test"]
    y_test = results["Y_test"]



    # Prepare statistics
    config = results["training_config"]
    unknown_label = config["unknown_label"]
    ordered_class_names = ordered_class_names + ["Unknown"]

    stats = {
        "image_size": IMAGE_RESIZE,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "class_names": ordered_class_names,
        "training_config": config,
        "models": {}
    }


    # Evaluate
    for name in ["SVC", "KNN"]:
        # --- General evaluation ---
        preds = results[name]["predictions"]
        acc = results[name]["accuracy"]
        train_time = results[name]["train_time_sec"]

        total_samples = len(preds)
        num_rejected = int((preds == unknown_label).sum())
        num_accepted = int((preds != unknown_label).sum())

        rejection_rate = num_rejected / total_samples
        acceptance_rate = num_accepted / total_samples

        print(f"\n=== {name} RESULTS ===")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Training Time: {train_time:.2f} sec")
        print(f"Rejected: {num_rejected} ({rejection_rate * 100:.2f}%)")
        print(f"Accepted: {num_accepted} ({acceptance_rate * 100:.2f}%)")

        # --- Known classes only evaluation ---
        mask = preds != unknown_label
        if mask.sum() == 0:
            print("No known predictions to evaluate (all samples rejected).")
        else:
            print("\nClassification Report (excluding rejection):")
            print(classification_report(
                y_test[mask],
                preds[mask],
                target_names=ordered_class_names[:-1]
            ))

        stats["models"][name] = {
            "accuracy": float(acc),
            "train_time_sec": float(train_time),
            "rejection": {
                "num_rejected": num_rejected,
                "num_accepted": num_accepted,
                "rejection_rate": float(rejection_rate),
                "acceptance_rate": float(acceptance_rate)
            }
        }

    # Save everything
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(DEST_FOLDER, f"run_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # Models
    joblib.dump(results["SVC"]["model"], os.path.join(save_path, "svm_model.pkl"))
    joblib.dump(results["KNN"]["model"], os.path.join(save_path, "knn_model.pkl"))
    joblib.dump(results["scaler"], os.path.join(save_path, "scaler.pkl"))
    joblib.dump(results["pca"], os.path.join(save_path, "pca.pkl"))

    # Statistics .json file
    with open(os.path.join(save_path, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\nAll models and statistics saved to:")
    print(save_path)
    print("\n===== PIPELINE COMPLETE =====")

if __name__ == "__main__":
    run_full_pipeline()
