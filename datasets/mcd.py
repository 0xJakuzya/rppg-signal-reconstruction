from pathlib import Path
from typing import Iterator
import numpy as np
from datasets.base import DatasetReader, Record

PPG_COLUMN = 0

class MCDReader(DatasetReader):
    """MCD-rPPG layout: flat video/ and ppg_sync/ folders keyed by filename stem"""
    
    def __init__(self, root: Path, camera: str = "FullHDwebcam"):
        self.root = Path(root)
        self.camera = camera

    def video_paths(self) -> list[Path]:
        return sorted((self.root / "video").glob(f"*_{self.camera}_*.avi"))

    @staticmethod
    def subject_of(video_path: Path) -> str:
        return video_path.stem.split("_", 1)[0]

    def subject_ids(self) -> list[str]:
        return sorted({self.subject_of(p) for p in self.video_paths()})

    def iter_records(self) -> Iterator[Record]:
        for video_path in self.video_paths():
            ppg_path = self.root / "ppg_sync" / f"{video_path.stem}.txt"
            if not ppg_path.exists():
                continue
            yield Record(
                subject_id=self.subject_of(video_path),
                clip_id=video_path.stem,
                video_path=video_path,
                load_ppg=self.make_ppg_loader(ppg_path),
            )

    @staticmethod
    def make_ppg_loader(ppg_path: Path):
        def load_ppg(frame_count: int, fps: float) -> np.ndarray:
            signal = np.loadtxt(ppg_path, usecols=PPG_COLUMN).astype(np.float32)
            if len(signal) >= frame_count:
                return signal[:frame_count]
            padded = np.zeros(frame_count, dtype=np.float32)
            padded[: len(signal)] = signal
            return padded

        return load_ppg
