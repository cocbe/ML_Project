import os
import cv2
import random
import shutil
import numpy as np
from count import get_images_in_ram 

aug_flag = False


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ================= CONFIG =================
AUGMENTATION = ['rotate','hflip','vflip','brightness','contrast','scale','crop', 'blur', 'hue']
OUTPUT_SUFFIX = "_aug"
# =========================================


def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def flip_image(image, flip_code=1):
    return cv2.flip(image, flip_code)

def change_brightness_contrast(image, brightness=0, contrast=0):
    img = image.astype(np.float32)
    img = img * (1 + contrast/100.0) + brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def scale_image(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h*scale_factor), int(w*scale_factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def random_crop(image, crop_ratio=0.1):
    h, w = image.shape[:2]
    crop_h, crop_w = int(h*crop_ratio), int(w*crop_ratio)
    start_x = random.randint(0, crop_w)
    start_y = random.randint(0, crop_h)
    end_x = w - (crop_w - start_x)
    end_y = h - (crop_h - start_y)
    cropped = image[start_y:end_y, start_x:end_x]
    return cv2.resize(cropped, (w, h))

def gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, kernel_size, sigmaX)

def adjust_hue(image, hue_delta):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,0] = (hsv[:,:,0] + hue_delta) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_image(image):
    choice = random.choice(AUGMENTATION)
    aug_name = ""
    
    if choice=='rotate':
        angle = random.uniform(-15,15)
        image = rotate_image(image,angle)
        aug_name = "rot"
    elif choice=='hflip':
        image = flip_image(image, flip_code=1)
        aug_name = "hflip"
    elif choice=='vflip':
        image = flip_image(image, flip_code=0)
        aug_name = "vflip"
    elif choice=='brightness':
        delta = random.uniform(-30,30)
        image = change_brightness_contrast(image,brightness=delta)
        aug_name = "brite"
    elif choice=='contrast':
        delta = random.uniform(-20,20)
        image = change_brightness_contrast(image,contrast=delta)
        aug_name = "cont"
    elif choice=='scale':
        factor = random.uniform(0.9,1.1)
        image = scale_image(image,factor)
        aug_name = "scale"
    elif choice=='crop':
        image = random_crop(image, crop_ratio=random.uniform(0.05,0.1))
        aug_name = "crop"
    elif choice=='blur':
        k_size = random.choice([3, 5]) 
        image = gaussian_blur(image, kernel_size=(k_size, k_size))
        aug_name = "blur"
    elif choice=='hue':
        delta = random.uniform(-10, 10) 
        image = adjust_hue(image, delta)
        aug_name = "hue"
        
    return image, aug_name



def balance_dataset(dataset_path, counts, output_path, target_count, save_aug, resize):
    global aug_flag 
    aug_flag = True 
    image_pixels_table_aug = {}

    # Delete old augmented folder if exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    current_image_pixels = get_images_in_ram() 

    for cls, current_count in counts.items():
        input_class_path = os.path.join(dataset_path, cls)
        output_class_path = os.path.join(output_path, cls)
        os.makedirs(output_class_path, exist_ok=True)

        images_in_ram = current_image_pixels.get(cls, []) 
        if not images_in_ram:
            print(f"Warning: Class '{cls}' has no images loaded in RAM. Skipping augmentation.")
            continue
             
        img_names = sorted([f for f in os.listdir(input_class_path) if f.lower().endswith(IMAGE_EXTENSIONS)])
        
        img_list_aug = list(images_in_ram) 
        
        needed_augmentations = target_count - len(images_in_ram) 
        needed_augmentations = max(0, needed_augmentations) 

        aug_count = 0
        while aug_count < needed_augmentations:
            idx = random.randint(0, len(images_in_ram) - 1)
            original_img = images_in_ram[idx]
            original_name = img_names[idx] 
            
            aug_img, aug_name = augment_image(original_img)
            
            img_list_aug.append(aug_img)
            
            if save_aug:
                base_name = os.path.splitext(original_name)[0]
                new_name = f"{base_name}_{aug_name}_{aug_count}.jpg"
                cv2.imwrite(os.path.join(output_class_path, new_name), aug_img)
            
            aug_count += 1

        image_pixels_table_aug[cls] = img_list_aug
        print(f"Class '{cls}': {len(img_list_aug)} images in RAM (Target: {target_count}), saved={save_aug}")

    return image_pixels_table_aug


def load_augmented_dataset(output_path, resize=(128,128)):
    image_pixels_table = get_images_in_ram()

    print(f"Loading augmented data from disk at: {output_path}")

    for cls in os.listdir(output_path):
        class_path = os.path.join(output_path, cls)
        if not os.path.isdir(class_path):
            continue

        image_pixels_table.setdefault(cls, [])
        img_list = image_pixels_table[cls]

        new_count = 0

        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            if "_" not in img_name:
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, resize)
            img_list.append(img)
            new_count += 1

        print(f"Loaded {new_count} augmented images for class '{cls}'")
        print(f"Class '{cls}' total in RAM: {len(img_list)}")

    return image_pixels_table

