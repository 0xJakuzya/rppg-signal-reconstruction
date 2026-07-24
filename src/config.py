from datetime import datetime
from pathlib import Path

# model: "physnet" | "factorizephys" | "ibvpnet" | "rhythmformer"
architecture = "physnet"

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
model_results_dir = results_dir / architecture
run_dir = model_results_dir / datetime.now().strftime("%d.%m.%Y")
val_patients_path = results_dir / "val_patients.json"
model_dir = run_dir
model_file_name = architecture
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

# factorizephys
fp_channels = [8, 12, 16]
fp_dropout = 0.1
fp_in_channels = 3
fp_use_fsam = True
fp_md_steps = 4
fp_md_rank = 2
fp_md_slices = 1
fp_md_residual = True
fp_align_channels = fp_channels[2] // 2

# ibvpnet
ibvp_channels = [8, 16, 24, 40, 64]
ibvp_in_channels = 3

# RhythmFormer
rf_embed_dims = [64, 64, 64]
rf_depths = [2, 2, 2]
rf_temporal_patches = [2, 4, 8]
rf_topks = [80, 80, 80]
rf_mlp_ratios = [1.5, 1.5, 1.5]
rf_head_dim = 16
rf_side_kernel = 3
rf_drop_path_rate = 0.0
rf_appearance_weight = 0.5
rf_difference_weight = 0.5

# train
epochs = 100
batch_sizes = {
    "physnet": 10,
    "factorizephys": 10,
    "ibvpnet": 10,
    "rhythmformer": 6,
}
batch_size = batch_sizes[architecture]
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
