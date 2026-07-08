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

def split_transfer_stages(files, val_patients=config.transfer_val_patients,
                          group_size=config.transfer_group_size,
                          stages=config.transfer_stages, seed=config.seed):
                          
    files_by_patient = {}
    for index, file in enumerate(files):
        files_by_patient.setdefault(patient_id(file), []).append(index)

    patients = sorted(files_by_patient)
    random.Random(seed).shuffle(patients)

    def indices_of(patient_list):
        indices = []
        for patient in patient_list:
            indices.extend(files_by_patient[patient])
        indices.sort()
        return indices

    val_indices = indices_of(patients[:val_patients])
    train_patients = patients[val_patients:]

    groups = []
    for stage in range(stages):
        stage_patients = train_patients[stage * group_size:(stage + 1) * group_size]
        if not stage_patients:
            break
        groups.append(indices_of(stage_patients))
    return val_indices, groups


def split_by_patient(files, val_split=config.val_split, seed=config.seed):
    files_by_patient = {}
    for index, file in enumerate(files):
        patient = patient_id(file)
        if patient not in files_by_patient:
            files_by_patient[patient] = []
        files_by_patient[patient].append(index)

    patients = sorted(files_by_patient)
    random.Random(seed).shuffle(patients)

    validation_count = int(len(patients) * val_split)
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, len(patients) - 1)
    validation_patients = patients[:validation_count]
    train_patients = patients[validation_count:]

    train_idx = []
    for patient in train_patients:
        for index in files_by_patient[patient]:
            train_idx.append(index)

    val_idx = []
    for patient in validation_patients:
        for index in files_by_patient[patient]:
            val_idx.append(index)

    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx

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

def build_dataloaders(dataset, train_idx, val_idx, batch_size, num_workers):
    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        shuffle=False,
    )
    return train_loader, validation_loader
