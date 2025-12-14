# Project README

## Key Scripts

- **`main.py`**  
  Runs the full pipeline:  
  **data augmentation → feature extraction → model training → evaluation → saving models**.  
  Also counts the dataset and loads images into memory.

- **`augment.py`**  
  Performs augmentation on the original dataset or loads already saved augmented images.  
  Augmentations include rotate, flip, scale, crop, blur, hue adjustments, etc.

- **`feature_extraction.py`**  
  Extracts features from images including **HOG**, **LBP**, **Color Histogram**, etc., used in `data_preparation.py`.

- **`model_training.py`**  
  Trains **SVM** and **KNN** models.  
  Handles **train/test split**, **feature scaling**, and **PCA**.

- **`data_preparation.py`**  
  Builds the feature matrix from images loaded in RAM.  
  Images are resized using the `IMAGE_RESIZE` value passed from `main.py`.

- **`live_app.py`**  
  Performs live inference (camera or folder mode).  
  Expects trained models and `run_stats.json` to be present in the `models/` directory.

- **`start_training.bat` / `start_live_app.bat`**  
  Convenience scripts for running training and live inference on **Windows**.

---

## Configuration

Each script contains **config variables** at the top of the file that can be edited easily  
(e.g., paths, image size, model selection, camera mode, etc.).

---

## Training Output

After training, all outputs (models, statistics, PCA, scaler, etc.) are saved in:
`saved_runs/`

You can select any trained run from `saved_runs/` and **copy its files into the `models/` folder**.  
These files will then be used as the **main models** by `live_app.py`.

---

## Live Application Modes

`live_app.py` supports **two modes**:

1. **Simulated camera mode**  
   Reads images sequentially from the `camera/` folder to simulate a camera stream.

2. **Live camera mode**  
   Uses a real camera device (webcam or connected camera).  
   **Recommendation:** Use [DroidCam](https://www.dev47apps.com/) to connect your mobile phone as a webcam if your PC has no camera.

The script also includes **configuration options** to choose **which trained model** (SVM or KNN) is used during live inference.
