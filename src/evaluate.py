import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.physnet import PhysNet
from src import config
from src.dataset import RPPGDataset, discover_windows
from src.postprocess import process_recordings
from src.utils import hr_metrics

def main(model_path):
    model_path = Path(model_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    files = discover_windows(config.processed_path)
    if not files:
        print(f"no windows found in {config.processed_path}")
        return
    
    print(f"eval dataset: {config.dataset} | windows: {len(files)}")

    model = PhysNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"model: PhysNet | weights: {model_path}")

    loader = DataLoader(RPPGDataset(files), batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False,)

    all_predictions = []

    with torch.no_grad():
        for frames, ppg in loader:
            frames = frames.to(device, non_blocking=True)
            predictions = model(frames).cpu().numpy()
            all_predictions.append(predictions)

    predictions = np.concatenate(all_predictions)
    processed = process_recordings(files, predictions)
    if processed is None:
        return
    predictions, targets, fps = processed
    max_shift = np.rint(np.asarray(fps) * config.max_shift_sec).astype(int)
    metrics = hr_metrics(predictions, targets, fps, max_shift=max_shift)

    print(f"HR MAE: {metrics['mae']:.2f} BPM | RMSE: {metrics['rmse']:.2f} BPM | corr: {metrics['corr']:.3f} | recordings: {metrics['n']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.evaluate <weights_path>")
    else:
        main(sys.argv[1])
