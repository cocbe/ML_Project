import numpy as np
import cv2
from feature_extraction import extract_features

def prepare_data(all_pixels,IMAGE_SIZE):
    print("\nPreparing the data")
    _dummy_image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    EXPECTED_VECTOR_LENGTH = extract_features(_dummy_image).size
    print(f"Expected feature vector length: {EXPECTED_VECTOR_LENGTH}")

    # Create label mapping
    class_names = sorted(all_pixels.keys())
    label_map = {name: i for i, name in enumerate(class_names)}

    print("Class-to-Label Mapping (Y Vector):")
    for name, label in label_map.items():
        print(f"  {name} -> {label}")

    # Feature Extraction
    X = []
    Y = []
    skipped = 0

    print("\nStarting feature extraction (HOG + LBP + Color + Edge + Saturation)...")

    for class_name in class_names:
        image_list = all_pixels[class_name]
        label = label_map[class_name]

        for i, image in enumerate(image_list):
            if image.shape[:2] != IMAGE_SIZE:
                image = cv2.resize(image, IMAGE_SIZE)

            try:
                feature_vector = extract_features(image)
                if feature_vector.size != EXPECTED_VECTOR_LENGTH:
                    print(
                        f"Skipping image in class '{class_name}' "
                        f"(index {i}): Expected {EXPECTED_VECTOR_LENGTH}, "
                        f"got {feature_vector.size}"
                    )
                    skipped += 1
                    continue

                X.append(feature_vector)
                Y.append(label)

            except Exception as e:
                print(
                    f"Error extracting features for class '{class_name}'"
                    f"(index {i}): {e}"
                )
                skipped += 1
                continue

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    print(f"\nFeature Matrix X shape: {X.shape}")
    print(f"Label Vector Y shape: {Y.shape}")
    print(f"Skipped images: {skipped}")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    print(f"Final Feature Matrix shape: {X.shape}")

    return X, Y, label_map