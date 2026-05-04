from __future__ import annotations

import ast
import dataclasses
import json
import os
import pickle
import random
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NpyArray:
    name: str
    descr: str
    shape: Tuple[int, ...]
    fortran_order: bool
    raw: bytes

    @property
    def itemsize(self) -> int:
        return _itemsize_from_descr(self.descr)

    @property
    def length(self) -> int:
        if not self.shape:
            return 1
        return self.shape[0]


@dataclass(slots=True)
class SequenceRecord:
    on_seq: str
    off_seq: str
    label: int
    reads: float | None = None
    dataset: str | None = None
    index: int | None = None


def _itemsize_from_descr(descr: str) -> int:
    if descr.startswith("<U") or descr.startswith(">U"):
        char_count = int(descr[2:])
        return char_count * 4
    if descr.endswith("1"):
        return 1
    if descr.endswith("2"):
        return 2
    if descr.endswith("4"):
        return 4
    if descr.endswith("8"):
        return 8
    raise ValueError(f"unsupported dtype descriptor: {descr}")


def _parse_npy_bytes(data: bytes, name: str) -> NpyArray:
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"{name} is not a .npy payload")
    major = data[6]
    if major == 1:
        header_len = struct.unpack("<H", data[8:10])[0]
        offset = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", data[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported .npy version: {major}")
    header_text = data[offset : offset + header_len].decode("latin1")
    header = ast.literal_eval(header_text)
    return NpyArray(
        name=name,
        descr=header["descr"],
        shape=tuple(header["shape"]),
        fortran_order=bool(header["fortran_order"]),
        raw=data[offset + header_len :],
    )


def load_npz_archive(path: str | os.PathLike[str], names: Iterable[str] | None = None) -> Dict[str, NpyArray]:
    selected = None if names is None else {name[:-4] if name.endswith(".npy") else name for name in names}
    archive: Dict[str, NpyArray] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for zip_name in zf.namelist():
            key = zip_name[:-4] if zip_name.endswith(".npy") else zip_name
            if selected is not None and key not in selected:
                continue
            archive[key] = _parse_npy_bytes(zf.read(zip_name), key)
    return archive


def _iter_int64_values(array: NpyArray) -> Iterator[int]:
    if array.descr not in {"<i8", "|i8"}:
        raise ValueError(f"expected int64 array, got {array.descr}")
    yield from (item[0] for item in struct.iter_unpack("<q", array.raw))


def _iter_float32_values(array: NpyArray) -> Iterator[float]:
    if array.descr not in {"<f4", "|f4"}:
        raise ValueError(f"expected float32 array, got {array.descr}")
    yield from (item[0] for item in struct.iter_unpack("<f", array.raw))


def _iter_unicode_values(array: NpyArray) -> Iterator[str]:
    if not (array.descr.startswith("<U") or array.descr.startswith(">U")):
        raise ValueError(f"expected unicode array, got {array.descr}")
    char_count = int(array.descr[2:])
    itemsize = char_count * 4
    for start in range(0, len(array.raw), itemsize):
        chunk = array.raw[start : start + itemsize]
        yield chunk.decode("utf-32le").rstrip("\x00")


def decode_uint8_matrix_sample(array: NpyArray, sample_index: int = 0) -> List[List[int]]:
    if array.descr != "|u1":
        raise ValueError(f"expected uint8 array, got {array.descr}")
    if len(array.shape) != 3:
        raise ValueError("expected a 3D tensor")
    sample_count, length, width = array.shape
    if sample_index < 0 or sample_index >= sample_count:
        raise IndexError("sample_index out of range")
    start = sample_index * length * width
    sample = array.raw[start : start + length * width]
    rows: List[List[int]] = []
    for row_start in range(0, len(sample), width):
        rows.append(list(sample[row_start : row_start + width]))
    return rows


def iter_sequence_records(
    path: str | os.PathLike[str],
    dataset: str | None = None,
    limit: int | None = None,
) -> Iterator[SequenceRecord]:
    archive = load_npz_archive(path, names={"on_seq", "off_seq", "y", "reads"})
    on_seq = archive["on_seq"]
    off_seq = archive["off_seq"]
    labels = archive["y"]
    reads = archive.get("reads")

    on_iter = _iter_unicode_values(on_seq)
    off_iter = _iter_unicode_values(off_seq)
    label_iter = _iter_int64_values(labels)
    read_iter = _iter_float32_values(reads) if reads is not None else None

    total = labels.length
    for index in range(total):
        if limit is not None and index >= limit:
            break
        record = SequenceRecord(
            on_seq=next(on_iter),
            off_seq=next(off_iter),
            label=int(next(label_iter)),
            reads=float(next(read_iter)) if read_iter is not None else None,
            dataset=dataset,
            index=index,
        )
        yield record


def collect_balanced_records(
    path: str | os.PathLike[str],
    dataset: str | None = None,
    positive_cap: int | None = None,
    negative_cap: int | None = None,
    seed: int = 42,
) -> List[SequenceRecord]:
    rng = random.Random(seed)
    pos_bucket: List[SequenceRecord] = []
    neg_bucket: List[SequenceRecord] = []
    pos_seen = 0
    neg_seen = 0

    for record in iter_sequence_records(path, dataset=dataset):
        bucket = pos_bucket if record.label == 1 else neg_bucket
        cap = positive_cap if record.label == 1 else negative_cap
        if record.label == 1:
            pos_seen += 1
            seen = pos_seen
        else:
            neg_seen += 1
            seen = neg_seen
        if cap is None:
            bucket.append(record)
            continue
        if len(bucket) < cap:
            bucket.append(record)
        else:
            index = rng.randrange(seen)
            if index < cap:
                bucket[index] = record

    sampled = pos_bucket + neg_bucket
    rng.shuffle(sampled)
    return sampled


def split_71515(
    records: Sequence[SequenceRecord],
    seed: int = 42,
) -> Tuple[List[SequenceRecord], List[SequenceRecord], List[SequenceRecord]]:
    rng = random.Random(seed)
    positives = [record for record in records if record.label == 1]
    negatives = [record for record in records if record.label == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def _split_class(items: List[SequenceRecord]) -> Tuple[List[SequenceRecord], List[SequenceRecord], List[SequenceRecord]]:
        n = len(items)
        train_end = int(round(n * 0.70))
        val_end = train_end + int(round(n * 0.15))
        train = items[:train_end]
        val = items[train_end:val_end]
        test = items[val_end:]
        return train, val, test

    pos_train, pos_val, pos_test = _split_class(positives)
    neg_train, neg_val, neg_test = _split_class(negatives)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
