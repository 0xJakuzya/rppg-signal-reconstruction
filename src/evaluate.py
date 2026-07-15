import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src import config
from src.dataset import RPPGDataset, discover_windows
from src.train import build_model
from src.utils import hr_metrics


def main(model_path=None):
    model_path = Path(model_path) if model_path else config.eval_model_path
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    fps = config.fps_target
    
    files = discover_windows(config.eval_windows_dir)
    print(f"eval dataset: {config.eval_dataset} | windows: {len(files)}")

    model = build_model(config.architecture).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"model: {config.architecture} | weights: {model_path}")

    loader = DataLoader(
        RPPGDataset(files),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for frames, ppg in loader:
            frames = frames.to(device, non_blocking=True)
            predictions = model(frames).cpu().numpy()
            all_predictions.append(predictions)
            all_targets.append(ppg.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    metrics = hr_metrics(predictions, targets, fps)
    print(f"HR MAE: {metrics['mae']:.2f} BPM | RMSE: {metrics['rmse']:.2f} BPM "
          f"| n={metrics['n']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
