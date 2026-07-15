from pathlib import Path
import cv2
import numpy as np
from datasets import get_dataset_reader
from src import config
from src.face_detector import FaceDetector
from src.utils import bandpass_filter, detrend, normalize_signal


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

def standardize_face_window(frames):
    frames = frames.astype(np.float32)
    if frames.max(initial=0.0) > 2.0:
        frames = frames / 255.0
    mean = frames.mean(axis=(0, 1, 2), keepdims=True)
    standard_deviation = frames.std(axis=(0, 1, 2), keepdims=True)
    normalized = (frames - mean) / (standard_deviation + config.eps)
    return normalized.astype(np.float32)

def prepare_frames(frames):
    return standardize_face_window(frames)

def prepare_label(ppg, fps):
    ppg = detrend(ppg)
    if len(ppg) >= int(fps * 2):
        ppg = bandpass_filter(ppg, fps)
    return normalize_signal(ppg)

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
    frames = frames[:length]
    ppg = ppg[:length]
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    start = 0
    while start <= length - config.window:
        end = start + config.window
        window_frames = prepare_frames(frames[start:end])
        window_ppg = prepare_label(ppg[start:end], fps)
        np.savez_compressed(
            out_dir / f"{stem}_{start:05d}.npz",
            frames=window_frames,
            ppg=window_ppg,
        )
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
    out_dir = Path(config.windows_dir) / record.subject_id
    return save_windows(frames, ppg, out_dir, record.clip_id, fps)


def main():
    windows_root = Path(config.windows_dir)
    reader = get_dataset_reader(config.train_dataset, config.dataset_root)
    detector = FaceDetector()
    total = 0
    try:
        for record in reader.iter_records():
            written = preprocess_record(record, detector)
            total += written
            if written:
                print(f"{record.clip_id}: {written} windows @ {record.fps:.2f} fps")
    finally:
        detector.close()
    print(f"done. {total} windows -> {windows_root}")


if __name__ == "__main__":
    main()
