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

# model: "physnet" | "factorizephys"
architecture = "factorizephys"

# paths
project_root = Path(__file__).resolve().parents[1]
dataset_root = dataset_roots[train_dataset]
windows_dir = windows_dirs[train_dataset]
results_dir = project_root / "result"
face_model_path = project_root / "face_landmarker.task"
model_results_dir = results_dir / architecture
run_dir = model_results_dir / datetime.now().strftime("%d.%m.%Y")
val_patients_path = model_results_dir / "val_patients.json"
model_dir = run_dir
model_file_name = architecture
history_file_name = "history.json"

# runtime
seed = 42
device = "cuda"
fps_target = 30.0
eps = 1e-6

# windowing
window = 300
stride = 150
frame_step = 1
video_fps = 30.0

# preprocessing filters
detrend_lambda = 100.0
band_lo_hz = 0.7
band_hi_hz = 2.5
cheby_order = 2
cheby_rs = 40.0
hr_lo_hz = 0.75
hr_hi_hz = 2.5

# face crop
face_size = 72
face_margin_frac = 0.5

# factorizephys
fp_channels = [8, 12, 16]
fp_dropout = 0.1
fp_in_channels = 3
fp_use_fsam = True
fp_md_steps = 4
fp_md_rank = 1
fp_md_slices = 1
fp_md_residual = True
fp_align_channels = fp_channels[2] // 2

# train
epochs = 30
batch_size = 10
lr = 3e-4
weight_decay = 1e-4
grad_clip = 1.0
warmup_epochs = 2
min_lr = 1e-6
max_shift_sec = 0.33
val_split = 0.1
num_workers = 4
early_stop_patience = 5

# evaluation (cross-dataset hold-out)
eval_windows_dir = windows_dirs[eval_dataset]
eval_model_path = run_dir / f"{model_file_name}_best.pth"
