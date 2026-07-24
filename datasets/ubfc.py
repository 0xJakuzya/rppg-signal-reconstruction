from pathlib import Path
from typing import Iterator
import numpy as np
from datasets.base import DatasetReader, Record

UBFC_FPS = 30.0

class UBFCReader(DatasetReader):
    """UBFC-rPPG layout: subjectN/{vid.avi, ground_truth.txt}."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def subject_dirs(self) -> list[Path]:
        return sorted(self.root.glob("subject*"))

    def subject_ids(self) -> list[str]:
        return [d.name for d in self.subject_dirs()]

    def iter_records(self) -> Iterator[Record]:
        for subject_dir in self.subject_dirs():
            video_path = subject_dir / "vid.avi"
            ground_truth_path = subject_dir / "ground_truth.txt"
            if not video_path.exists() or not ground_truth_path.exists():
                continue
            yield Record(
                subject_id=subject_dir.name,
                clip_id=subject_dir.name,
                video_path=video_path,
                fps=UBFC_FPS,
                load_ppg=self.make_ppg_loader(ground_truth_path),
            )

    @staticmethod
    def make_ppg_loader(ground_truth_path: Path):
        def load_ppg(frame_count: int, fps: float) -> np.ndarray:
            ground_truth = np.loadtxt(ground_truth_path).astype(np.float32)
            trace, times = ground_truth[0], ground_truth[2]
            order = np.argsort(times)
            times = times[order] - times[order][0]
            trace = trace[order]
            target_times = np.arange(frame_count, dtype=np.float32) / fps
            return np.interp(target_times, times, trace).astype(np.float32)
        return load_ppg
