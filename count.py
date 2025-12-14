import os
import cv2

image_pixels_table = {}  # {class_name: [numpy arrays]}

def count_images(dataset_path, resize):
    global image_pixels_table
    counts = {}
    image_pixels_table = {}

    for cls in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(class_path):
            continue

        images = sorted([f for f in os.listdir(class_path) if f.lower().endswith(".jpg")])
        counts[cls] = len(images)

        
        img_list = []
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, resize)
            img_list.append(img)

        image_pixels_table[cls] = img_list
    return counts

def get_images_in_ram():
    global image_pixels_table
    return image_pixels_table
