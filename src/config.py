from datetime import datetime
from pathlib import Path

# datasets
train_dataset = "mcd"
eval_dataset = "ubfc"
dataset_roots = {
    "mcd": Path("D:/datasets/mcd_rppg"),
    "ubfc": Path("D:/datasets/UBFC-rPPG"),
}
windows_dirs = {
    "mcd": Path("C:/datasets/data/mcd_processed_30fps_w300"),
    "ubfc": Path("D:/datasets/data/ubfc_processed"),
}

# paths
project_root = Path(__file__).resolve().parents[1]
dataset_root = dataset_roots[train_dataset]
windows_dir = windows_dirs[train_dataset]
results_dir = project_root / "result"
face_model_path = project_root / "face_landmarker.task"
model_results_dir = results_dir / "physnet"
run_dir = model_results_dir / datetime.now().strftime("%d.%m.%Y")
val_patients_path = results_dir / "val_patients.json"
model_dir = run_dir
model_file_name = "physnet"
history_file_name = "history.json"

# preprocessing
cameras = ["USBVideo"]
skip_existing = True
window = 300
stride = 150
frame_step = 1
video_fps = 30.0
detrend_lambda = 100.0
band_lo_hz = 0.7
band_hi_hz = 2.5
cheby_order = 2
cheby_rs = 40.0
hr_lo_hz = 0.75
hr_hi_hz = 2.5
face_size = 72
face_margin_frac = 0.5

# runtime
seed = 42
device = "cuda"
fps_target = 30.0
eps = 1e-6
color_order = "rgb"

# train
epochs = 100
batch_size = 10
lr = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
warmup_epochs = 0
min_lr = 1e-6
max_shift_sec = 0.33
val_split = 0.1
num_workers = 4
early_stop_patience = 5

# evaluation
eval_windows_dir = windows_dirs[eval_dataset]
eval_model_path = run_dir / f"{model_file_name}_best.pth"
