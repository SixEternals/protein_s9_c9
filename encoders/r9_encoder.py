from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from utils.sequence import encode_r9_matrix, normalize_sequence


@dataclass
class R9Encoder:
    length: int = 23

    def encode_pair(self, on_seq: str, off_seq: str) -> List[List[int]]:
        return encode_r9_matrix(on_seq, off_seq)

    def encode_batch(self, pairs: Sequence[tuple[str, str]]) -> List[List[List[int]]]:
        return [self.encode_pair(on_seq, off_seq) for on_seq, off_seq in pairs]

    def normalize_pair(self, on_seq: str, off_seq: str) -> tuple[str, str]:
        return normalize_sequence(on_seq, self.length), normalize_sequence(off_seq, self.length)

