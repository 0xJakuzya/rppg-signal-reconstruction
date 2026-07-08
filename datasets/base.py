from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
import numpy as np

@dataclass
class Record:
    subject_id: str
    clip_id: str
    video_path: Path
    load_ppg: Callable[[int, float], np.ndarray]

class DatasetReader(ABC):

    @abstractmethod
    def iter_records(self) -> Iterator[Record]:
        """Yield one Record per video to be turned into training windows."""

    @abstractmethod
    def subject_ids(self) -> list[str]:
        """All subject ids in the dataset, sorted and de-duplicated."""
