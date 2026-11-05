from pathlib import Path
from typing import Iterator
import numpy as np
from datasets.base import DatasetReader, Record

ppg_column = 0
default_cameras = ["FullHDwebcam", "IriunWebcam", "USBVideo"]
default_states = ["before", "after"]

class MCDReader(DatasetReader):

    def __init__(self, root: Path, cameras: list[str] = None, states: list[str] = None):
        self.root = Path(root)
        self.cameras = list(cameras) if cameras else list(default_cameras)
        self.states = list(states) if states else list(default_states)

    def video_paths(self) -> list[Path]:
        paths = []
        for camera in self.cameras:
            for state in self.states:
                for video_path in (self.root / "video").glob(f"*_{camera}_{state}.avi"):
                    if self.is_valid_name(video_path):
                        paths.append(video_path)
        return sorted(paths)

    def is_valid_name(self, video_path: Path) -> bool:
        parts = video_path.stem.split("_")
        if len(parts) != 3:
            return False
        subject, camera, state = parts
        if not subject.isdigit():
            return False
        return camera in self.cameras and state in self.states

    @staticmethod
    def subject_of(video_path: Path) -> str:
        return video_path.stem.split("_", 1)[0]

    def subject_ids(self) -> list[str]:
        return sorted({self.subject_of(p) for p in self.video_paths()})

    def read_fps(self, stem: str) -> float:
        meta_path = self.root / "meta" / f"{stem}.txt"
        if not meta_path.exists():
            return None

        timestamps = []
        with open(meta_path, encoding="utf-8") as meta_file:
            for line in meta_file:
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    continue
                timestamps.append(np.datetime64(parts[1].strip().replace(" ", "T")))

        if len(timestamps) < 2:
            return None
        duration = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "s")
        if duration <= 0:
            return None
        return (len(timestamps) - 1) / duration

    def iter_records(self) -> Iterator[Record]:
        for video_path in self.video_paths():
            stem = video_path.stem
            ppg_path = self.root / "ppg_sync" / f"{stem}.txt"
            if not ppg_path.exists():
                continue

            fps = self.read_fps(stem)
            if fps is None:
                print(f"skip {stem}: no usable timestamps in meta/")
                continue

            yield Record(
                subject_id=self.subject_of(video_path),
                clip_id=stem,
                video_path=video_path,
                fps=fps,
                load_ppg=self.make_ppg_loader(ppg_path),
            )

    @staticmethod
    def make_ppg_loader(ppg_path: Path):
        def load_ppg(frame_count: int, fps: float) -> np.ndarray:
            signal = np.loadtxt(ppg_path, usecols=ppg_column).astype(np.float32)
            if len(signal) >= frame_count:
                return signal[:frame_count]
            padded = np.zeros(frame_count, dtype=np.float32)
            padded[: len(signal)] = signal
            return padded

        return load_ppg
