from pathlib import Path
import numpy as np
from src.utils import bandpass_filter, detrend, normalize_signal

def merge_windows(windows, starts):
    if not len(windows) or len(windows) != len(starts):
        print("each window must have a start")
        return None

    if any(start < 0 for start in starts):
        print("window start must be non-negative")
        return None

    length = max(start + len(window) for window, start in zip(windows, starts))
    signal = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)

    for window, start in zip(windows, starts):
        end = start + len(window)
        signal[start:end] += window
        counts[start:end] += 1

    if np.any(counts == 0):
        print("gaps between windows")
        return None

    return signal / counts

def restore_waveform(difference):
    waveform = np.zeros(len(difference), dtype=np.float64)
    waveform[1:] = np.cumsum(difference[:-1], dtype=np.float64)
    return waveform

def postprocess_signal(signal, fps):
    signal = detrend(signal)
    signal = bandpass_filter(signal, fps)
    if signal is None:
        return None
    return normalize_signal(signal)

def process_recordings(files, predictions):
    if not len(files) or len(files) != len(predictions):
        print("each window must have one prediction")
        return None

    recordings = {}
    for file, prediction in zip(files, predictions):
        path = Path(file)
        clip_id = path.stem.rsplit("_", 1)[0]
        with np.load(path) as sample:
            if not {"ppg_raw", "fps", "start"}.issubset(sample.files):
                print(f"{path}: regenerate windows with ppg_raw, fps and start")
                return None
            fps = float(sample["fps"].item())
            start = int(sample["start"].item())
            target = sample["ppg_raw"].astype(np.float64)

        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.ndim != 1 or prediction.shape != target.shape or start < 0:
            print(f"{path}: invalid window shape or start")
            return None
        
        recording = recordings.setdefault((path.parent, clip_id), {"fps": fps, "windows": []})
        if not np.isclose(recording["fps"], fps):
            print(f"{clip_id}: inconsistent fps")
            return None
        
        recording["windows"].append((start, prediction, target))

    all_predictions = []
    all_targets = []
    all_fps = []

    for recording in recordings.values():
        fps = recording["fps"]
        starts, predictions, targets = zip(*recording["windows"])
        difference = merge_windows(predictions, starts)
        ppg_raw = merge_windows(targets, starts)
        
        if difference is None or ppg_raw is None:
            return None
        
        prediction = postprocess_signal(restore_waveform(difference), fps)
        target = postprocess_signal(ppg_raw, fps)
        
        if prediction is None or target is None:
            return None

        all_predictions.append(prediction)
        all_targets.append(target)
        all_fps.append(fps)

    return all_predictions, all_targets, all_fps
