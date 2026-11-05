from pathlib import Path
import cv2
import numpy as np
from datasets import get_dataset_reader
from src import config
from src.face_detector import FaceDetector

def missing_face():
    shape = (config.face_size, config.face_size, 3)
    return np.full(shape, np.nan, np.float32)

def fill_missing(frames):
    frame_count = len(frames)
    flat = frames.reshape(frame_count, -1)
    valid = np.isfinite(flat).all(axis=1)
    if not valid.any():
        return None
    indices = np.arange(frame_count)
    for column in range(flat.shape[1]):
        flat[:, column] = np.interp(indices, indices[valid], flat[valid, column])
    return flat.reshape(frames.shape).astype(np.float32)

def diffnormalize_data(frames):
    frames = frames.astype(np.float32)
    if len(frames) < 2:
        return np.zeros_like(frames)
    difference = (frames[1:] - frames[:-1]) / (frames[1:] + frames[:-1] + config.diffnormalized_epsilon)
    standard_deviation = difference.std()
    if standard_deviation > config.eps:
        difference = difference / standard_deviation
    else:
        difference.fill(0.0)
    normalized = np.zeros_like(frames)
    normalized[:-1] = difference
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized

def diffnormalize_label(ppg):
    ppg = np.asarray(ppg, dtype=np.float32)
    if len(ppg) < 2:
        return np.zeros_like(ppg)
    difference = np.diff(ppg)
    standard_deviation = difference.std()
    if standard_deviation > config.eps:
        difference = difference / standard_deviation
    else:
        difference.fill(0.0)
    normalized = np.zeros_like(ppg)
    normalized[:-1] = difference
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized

def extract_faces(video_path, detector):
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    missing = 0
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded % config.frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.landmarks(frame)
            if landmarks is None:
                frames.append(missing_face())
                missing += 1
            else:
                face = detector.face_crop(frame, landmarks, config.face_size).astype(np.float32)
                frames.append(face)
        decoded += 1
    capture.release()
    return np.asarray(frames, np.float32), missing

def save_windows(frames, ppg, out_dir, stem, fps):
    length = min(len(frames), len(ppg))
    frames = diffnormalize_data(frames[:length])
    ppg_raw = np.asarray(ppg[:length], dtype=np.float32)
    ppg = diffnormalize_label(ppg_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    start = 0
    while start <= length - config.window:
        end = start + config.window
        window_frames = frames[start:end]
        window_ppg = ppg[start:end]
        np.savez_compressed(out_dir / f"{stem}_{start:05d}.npz", frames=window_frames, ppg=window_ppg,
                            ppg_raw=ppg_raw[start:end], fps=fps, start=start)
        count += 1
        start += config.stride
    return count

def preprocess_record(record, detector):
    fps = record.fps / config.frame_step
    frames, missing = extract_faces(record.video_path, detector)
    if len(frames) == 0:
        print(f"skip {record.subject_id}: no frames")
        return 0

    missing_ratio = missing / len(frames)
    if missing_ratio > 0.2:
        print(f"skip {record.subject_id}: missing face ratio {missing_ratio:.2f}")
        return 0

    frames = fill_missing(frames)
    if frames is None:
        print(f"skip {record.subject_id}: no valid frames")
        return 0

    ppg = record.load_ppg(len(frames), fps)
    out_dir = Path(config.processed_path) / record.subject_id
    return save_windows(frames, ppg, out_dir, record.clip_id, fps)

def main():
    windows_root = Path(config.processed_path)
    reader_kwargs = {}
    if config.dataset == "mcd" and config.cameras:
        reader_kwargs["cameras"] = config.cameras # FullHD cameras 
    reader = get_dataset_reader(config.dataset, config.dataset_path, **reader_kwargs)
    detector = FaceDetector()
    total = 0
    skipped = 0
    try:
        for record in reader.iter_records():
            out_dir = windows_root / record.subject_id
            if config.skip_existing and any(out_dir.glob(f"{record.clip_id}_*.npz")):
                for file in out_dir.glob(f"{record.clip_id}_*.npz"):
                    with np.load(file) as sample:
                        if not {"ppg_raw", "fps", "start"}.issubset(sample.files):
                            print(f"{file}: regenerate windows in a new processed_path")
                            return
                skipped += 1
                continue
            written = preprocess_record(record, detector)
            total += written
            if written:
                print(f"{record.clip_id}: {written} windows @ {record.fps:.2f} fps")
    finally:
        detector.close()
    print(f"done. {total} new windows ({skipped} clips already present) -> {windows_root}")

if __name__ == "__main__":
    main()
