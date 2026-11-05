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

def split_train_val(files, val_split=config.val_split, seed=config.seed):
    files_by_patient = {}
    for index, file in enumerate(files):
        files_by_patient.setdefault(patient_id(file), []).append(index)
    present = sorted(files_by_patient)
    shuffled = list(present)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_split))
    val_patients = set(shuffled[:val_count])

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
    val_dataset = RPPGDataset(dataset.files)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    validation_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=batch_size, num_workers=num_workers, shuffle=False,)
    return train_loader, validation_loader

class RPPGDataset(Dataset):
    def __init__(self, files):
        self.files = []
        for file in files:
            self.files.append(str(file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as sample:
            frames = torch.from_numpy(sample["frames"]).float()
            ppg = torch.from_numpy(sample["ppg"]).float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames, ppg
