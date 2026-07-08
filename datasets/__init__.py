from pathlib import Path
from datasets.base import DatasetReader, Record
from datasets.mcd import MCDReader
from datasets.ubfc import UBFCReader

READERS = {
    "ubfc": UBFCReader,
    "mcd": MCDReader,
}

def get_dataset_reader(name: str, root: Path) -> DatasetReader:
    key = name.lower()
    if key not in READERS:
        raise ValueError(f"unknown dataset '{name}', expected one of {sorted(READERS)}")
    return READERS[key](root)

__all__ = ["DatasetReader", "Record", "UBFCReader", "MCDReader", "get_dataset_reader"]
