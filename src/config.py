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
val_patients_path = results_dir / "val_patients.json"

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

# data representation
data_type = "Standardized"
label_type = "Standardized"

# train
epochs = 20
batch_size = 10  
lr = 3e-4
mid_lr = 3e-5            
finetune_lr = 5e-6     
weight_decay = 1e-4
dropout = 0.2
grad_clip = 1.0

max_shift_sec = 0.33
val_split = 0.2
num_workers = 4
model_dir = results_dir
model_file_name = "physnet"
use_last_epoch = False
plot_losses_and_lr = True
early_stop_patience = 3

# evaluation (cross-dataset hold-out)
eval_windows_dir = windows_dirs[eval_dataset]
eval_model_path = results_dir / f"{model_file_name}_best.pth"