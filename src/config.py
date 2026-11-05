from datetime import datetime
from pathlib import Path

# path
project_root = Path(__file__).resolve().parents[1]
result_dir = project_root / "result"
face_model = project_root / "face_landmarker.task"
run_dir = result_dir / datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
dataset_path = Path("D:/datasets/UBFC-rPPG")
processed_path = Path("D:/datasets/data/ubfc_processed")

# preprocessing

## window
window = 300
stride = 150
frame_step = 1
fps = 30.0
diffnormalized_epsilon = 1e-7

## filtering
filter_type = "butterworth"
filter_order = 2
band_lo_hz = 0.5
band_hi_hz = 10.0
detrend_lambda = 100.0

# fourier transform 
hr_lo_hz = 0.7 # 42 BPM
hr_hi_hz = 3.0 # 180 BPM

## mediapipe settings
face_size = 72
face_margin_frac = 0.5

# runtime
seed = 42
device = "cuda"
fps_target = 30.0
eps = 1e-6
color = "rgb"

# train
epochs = 20
batch_size = 10
lr = 1e-3
weight_decay = 1e-4 
grad_clip = 1.0
warmup_epochs = 0
min_lr = 1e-6
val_split = 0.2
num_workers = 4
early_stop_patience = 4
skip_existing = True
