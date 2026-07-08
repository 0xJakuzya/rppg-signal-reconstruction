from pathlib import Path

# datasets
train_dataset = "mcd"
eval_dataset = "ubfc"
dataset_roots = {
    "mcd": Path("D:/datasets/mcd_rppg"),
    "ubfc": Path("D:/datasets/UBFC-rPPG"),
}
windows_dirs = {
    "mcd": Path("D:/datasets/data/mcd_processed_30fps"),
    "ubfc": Path("D:/datasets/data/ubfc_processed"),
}

# paths
project_root = Path(__file__).resolve().parents[1]
dataset_root = dataset_roots[train_dataset]
windows_dir = windows_dirs[train_dataset]
results_dir = project_root / "result"
face_model_path = project_root / "face_landmarker.task"

# runtime
seed = 42
device = "cuda"
fps_target = 30.0        
eps = 1e-6

# windowing
window = 256              
stride = 128               
frame_step = 1            
video_fps = 30.0        

# preprocessing filters 
detrend_lambda = 100.0
band_lo_hz = 0.5
band_hi_hz = 3.0
cheby_order = 2
cheby_rs = 40.0
hr_lo_hz = 0.5            
hr_hi_hz = 3.0

# face crop 
face_size = 72
face_margin_frac = 0.5

# data representation
data_type = "DiffNormalized"
label_type = "DiffNormalized"

# train
epochs = 50
batch_size = 10           
lr = 3e-4               
finetune_lr = 5e-6        
weight_decay = 1e-4      
dropout = 0.2             
grad_clip = 1.0          

max_shift_sec = 0.33       
val_split = 0.2
num_workers = 4       
model_path = "model.pth"
model_dir = results_dir
model_file_name = "physnet"
use_last_epoch = False
plot_losses_and_lr = True
early_stop_patience = 3    

# transfer learning
transfer_val_patients = 15     
transfer_group_size = 50       
transfer_stages = 4
resume_path = None           

# evaluation (cross-dataset hold-out)
eval_windows_dir = windows_dirs[eval_dataset]
eval_model_path = results_dir / f"{model_file_name}_best.pth"