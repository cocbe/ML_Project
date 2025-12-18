Project Team Roles & How to run

Overview
- This file maps the 5 team roles to code, explains where data belongs, and lists commands to run the pipeline and live app.

Dataset placement
- Put the original images in `dataset/` using one subfolder per class, e.g.:
  - dataset/Glass/*.jpg
  - dataset/Paper/*.jpg
  - dataset/Cardboard/*.jpg
  - dataset/Plastic/*.jpg
  - dataset/Metal/*.jpg
  - dataset/Trash/*.jpg
- The pipeline accepts `.jpg`, `.jpeg`, and `.png` files.

Augmented data
- Augmented images (if saved) are written to `dataset_aug/` by the augmentation pipeline.
- To use an existing augmented run, copy a `saved_runs/run_xxx` folder into `models/` or set `MODELS_FOLDER` accordingly in `live_app.py`.

Member assignments (files & responsibilities)
- Member 1 — Data Engineer (Augmentation & Preprocessing)
  - Files: `augment.py`, `count.py`, `data_preparation.py`, `dataset/`, `dataset_aug/`
  - Tasks: organize dataset into class folders; run augmentation via `main.py` which calls `balance_dataset()`; ensure target_count in `main.py` is set (e.g., ~500 per class).
  - Augmentation requirement: increase dataset size by at least 30% and balance classes. `balance_dataset()` currently produces images up to `TARGET_COUNT` and can save augmented samples to `dataset_aug/`.

- Member 2 — Feature Extraction Lead
  - Files: `feature_extraction.py`, `data_preparation.py`
  - Tasks: implement/adjust feature descriptors (HOG, LBP, color histograms are implemented); ensure output vectors are fixed-length and compatible with classifiers.

- Member 3 — SVM Model Architect
  - Files: `model_training.py`
  - Tasks: SVM is implemented using `sklearn.svm.SVC` with RBF kernel and probability outputs; unknown/rejection for SVM is performed via `svm_unknown_threshold` in `model_training.py` and stored in the training config.

- Member 4 — k-NN Model Architect & Analyst
  - Files: `model_training.py`
  - Tasks: k-NN implemented using `sklearn.neighbors.KNeighborsClassifier`; unknown/rejection uses a distance-based threshold derived from training distances. Tune `KNN_N_NEIGHBORS` and `KNN_UNKNOWN_THRESHOLD_STD` in `model_training.py`.

- Member 5 — Deployment & Integration Specialist
  - Files: `live_app.py`, `saved_runs/`, `models/`
  - Tasks: `live_app.py` loads models from `models/` (expects `svm_model.pkl`, `knn_model.pkl`, `scaler.pkl`, `pca.pkl`, `run_stats.json`). Use `saved_runs/` to store experiments; copy chosen run into `models/` for live usage.

How to run
- Train (full pipeline):
```powershell
python main.py
```
- Start live app (after copying a `saved_runs/run_xxx` into `models/`):
```powershell
python live_app.py
```
- To copy a saved run into `models/` (PowerShell):
```powershell
mkdir models
Copy-Item -Path saved_runs\run_xxx\* -Destination models -Recurse -Force
```

Notes / Next improvements
- Confirm augmentation increases dataset >=30%: `balance_dataset()` targets a per-class `TARGET_COUNT` in `main.py`.
- Add a sample `dataset` with one image per class for quick smoke tests.
- Consider adding a `requirements.txt` and a `run.sh` / `run.bat` script that automates standard commands.

If you want, I can:
- Add `requirements.txt` listing packages used.
- Add a sample `dataset/` layout and a simple smoke-test script.
- Create a short `report_template.md` for each member's write-up.
