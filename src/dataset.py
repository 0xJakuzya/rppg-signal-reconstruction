import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src import config

def discover_windows(data_dir):
    files = []
    for path in Path(data_dir).rglob("*.npz"):
        files.append(str(path))
    files.sort()
    return files

def patient_id(file):
    stem = Path(file).stem
    return stem.split("_", 1)[0]

def load_val_patients(path=config.val_patients_path):
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    return None

def save_val_patients(patients, path=config.val_patients_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(patients), f, indent=2)

def split_train_val(files, val_split=config.val_split, seed=config.seed):
    files_by_patient = {}
    for index, file in enumerate(files):
        files_by_patient.setdefault(patient_id(file), []).append(index)
    present = sorted(files_by_patient)
    val_patients = load_val_patients()
    if val_patients is None:
        shuffled = list(present)
        random.Random(seed).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_split)) # 0.1% -> 60 patients
        val_patients = set(shuffled[:val_count])
        save_val_patients(val_patients)
        print(f"Fixed validation set created: {len(val_patients)} patients")

    def indices_of(patient_list):
        indices = []
        for patient in patient_list:
            indices.extend(files_by_patient[patient])
        indices.sort()
        return indices

    val_present = [p for p in present if p in val_patients]
    train_present = [p for p in present if p not in val_patients]
    return indices_of(train_present), indices_of(val_present)

def build_dataloaders(dataset, train_idx, val_idx, batch_size, num_workers):
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    val_dataset = RPPGDataset(
        dataset.files,
        crop=dataset.crop,
        random_crop=False,
        color_order=dataset.color_order,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        shuffle=False,
    )
    
    return train_loader, validation_loader

class RPPGDataset(Dataset):
    def __init__(self, files, crop="config", random_crop="config", color_order="config"):
        self.files = []
        for file in files:
            self.files.append(str(file))
        self.crop = config.crop_window if crop == "config" else crop
        self.random_crop = config.crop_random if random_crop == "config" else random_crop
        self.color_order = (config.color_order if color_order == "config" else color_order).lower()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as sample:
            frames = torch.from_numpy(sample["frames"]).float()
            ppg = torch.from_numpy(sample["ppg"]).float()
        start, end = self.crop_bounds(frames.shape[0])
        frames = frames[start:end]
        ppg = ppg[start:end]
        if self.color_order == "rgb":
            frames = frames[..., [2, 1, 0]]
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames, ppg
