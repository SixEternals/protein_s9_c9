from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


BASE_ALPHABET = ("A", "T", "G", "C")
PAIR_ALPHABET = ("A", "T", "G", "C", "-")


@dataclass(frozen=True)
class R9EncoderLayout:
    base_offset: int = 0
    event_offset: int = 5
    region_offset: int = 7
    width: int = 9


@dataclass(frozen=True)
class C9EncoderLayout:
    base_offset: int = 0
    event_offset: int = 5
    state_offset: int = 7
    width: int = 9


def normalize_sequence(seq: str, length: int = 23) -> str:
    seq = (seq or "").strip().upper().replace("U", "T")
    cleaned: List[str] = []
    for ch in seq:
        if ch in {"A", "T", "G", "C", "N", "-"}:
            cleaned.append(ch)
        else:
            cleaned.append("N")
    normalized = "".join(cleaned)
    if len(normalized) < length:
        normalized = normalized + ("-" * (length - len(normalized)))
    elif len(normalized) > length:
        normalized = normalized[:length]
    return normalized


def validate_sequence(seq: str, name: str = "sequence", length: int = 23) -> None:
    """Strictly validate a biological sequence for prediction input.

    Raises ValueError with a clear message if the sequence is invalid.
    Rules:
      - Must not be empty
      - Must be exactly `length` characters
      - Only A, T, C, G, U allowed (U will be converted to T by caller)
      - No N, -, or other placeholders allowed for prediction input
    """
    if not seq:
        raise ValueError(f"{name} is empty")
    seq = seq.strip().upper()
    if len(seq) != length:
        raise ValueError(f"{name} length must be {length}, got {len(seq)}")
    valid = set("ATCGU")
    invalid = [ch for ch in seq if ch not in valid]
    if invalid:
        raise ValueError(f"{name} contains invalid characters: {set(invalid)}; only A/T/C/G/U allowed")


def canonical_pair(on_base: str, off_base: str) -> Tuple[str, str]:
    on_base = _normalize_base(on_base)
    off_base = _normalize_base(off_base)

    if on_base == "N" and off_base in BASE_ALPHABET:
        on_base = off_base
    if off_base == "N" and on_base in BASE_ALPHABET:
        off_base = on_base
    if on_base == "N" and off_base == "N":
        return "-", "-"
    return on_base, off_base


def _normalize_base(ch: str) -> str:
    ch = (ch or "-").upper()
    if ch == "U":
        return "T"
    if ch in {"A", "T", "G", "C", "N", "-"}:
        return ch
    return "N"


def base_union_bits(on_base: str, off_base: str) -> List[int]:
    on_base, off_base = canonical_pair(on_base, off_base)
    bits = [0, 0, 0, 0, 0]
    for base in {on_base, off_base}:
        if base == "A":
            bits[0] = 1
        elif base == "T":
            bits[1] = 1
        elif base == "G":
            bits[2] = 1
        elif base == "C":
            bits[3] = 1
        elif base == "-":
            bits[4] = 1
    return bits


def event_name_from_pair(on_base: str, off_base: str) -> str:
    on_base, off_base = canonical_pair(on_base, off_base)
    if on_base == off_base:
        return "match"
    if on_base == "-" and off_base != "-":
        return "insertion"
    if off_base == "-" and on_base != "-":
        return "deletion"
    return "mismatch"


def event_bits_from_name(name: str) -> Tuple[int, int]:
    mapping = {
        "match": (0, 0),
        "insertion": (0, 1),
        "mismatch": (1, 0),
        "deletion": (1, 1),
    }
    if name not in mapping:
        raise ValueError(f"unknown event name: {name}")
    return mapping[name]


def event_name_from_bits(bits: Sequence[int]) -> str:
    pair = tuple(int(value) for value in bits[:2])
    mapping = {
        (0, 0): "match",
        (0, 1): "insertion",
        (1, 0): "mismatch",
        (1, 1): "deletion",
    }
    if pair not in mapping:
        raise ValueError(f"unknown event bits: {pair}")
    return mapping[pair]


def event_bits_from_pair(on_base: str, off_base: str) -> Tuple[int, int]:
    return event_bits_from_name(event_name_from_pair(on_base, off_base))


def region_bits(position_1_based: int) -> Tuple[int, int]:
    if position_1_based <= 15:
        return (0, 1)
    if position_1_based <= 20:
        return (1, 0)
    return (0, 0)


def continuous_state_bits(state: int) -> Tuple[int, int]:
    if state <= 0:
        return (0, 0)
    if state == 1:
        return (0, 1)
    if state == 2:
        return (1, 0)
    return (1, 1)


def encode_r9_matrix(on_seq: str, off_seq: str) -> List[List[int]]:
    on_seq = normalize_sequence(on_seq)
    off_seq = normalize_sequence(off_seq)
    rows: List[List[int]] = []
    for idx, (on_base, off_base) in enumerate(zip(on_seq, off_seq), start=1):
        on_base, off_base = canonical_pair(on_base, off_base)
        row = []
        row.extend(base_union_bits(on_base, off_base))
        row.extend(event_bits_from_pair(on_base, off_base))
        row.extend(region_bits(idx))
        rows.append(row)
    return rows


def encode_c9_matrix(on_seq: str, off_seq: str) -> List[List[int]]:
    on_seq = normalize_sequence(on_seq)
    off_seq = normalize_sequence(off_seq)
    pairs = [canonical_pair(o, d) for o, d in zip(on_seq, off_seq)]
    event_names = [event_name_from_pair(o, d) for o, d in pairs]
    mismatch_mask = [name == "mismatch" for name in event_names]
    state_codes = _continuous_state_codes(mismatch_mask)

    rows: List[List[int]] = []
    for (on_base, off_base), event_name, state in zip(pairs, event_names, state_codes):
        row = []
        row.extend(base_union_bits(on_base, off_base))
        row.extend(event_bits_from_name(event_name))
        row.extend(continuous_state_bits(state))
        rows.append(row)
    return rows


def _continuous_state_codes(mask: Sequence[bool]) -> List[int]:
    codes = [0] * len(mask)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        run_len = j - i
        if run_len == 1:
            code = 1
        elif run_len == 2:
            code = 2
        else:
            code = 3
        for k in range(i, j):
            codes[k] = code
        i = j
    return codes


def mismatch_mask_from_matrix(matrix: Sequence[Sequence[int]]) -> List[bool]:
    mask: List[bool] = []
    for row in matrix:
        mask.append(bool(row[5] or row[6]))
    return mask


def c9_mask_from_matrix(matrix: Sequence[Sequence[int]]) -> List[bool]:
    mask: List[bool] = []
    for row in matrix:
        mask.append(bool(row[5] == 1 and row[6] == 0))
    return mask


def run_length_stats(mask: Sequence[bool]) -> Tuple[int, int, int, List[int]]:
    runs: List[int] = []
    current = 0
    transitions = 0
    longest = 0
    for idx, value in enumerate(mask):
        if value:
            current += 1
            if idx > 0 and mask[idx - 1]:
                transitions += 1
        elif current:
            runs.append(current)
            if current > longest:
                longest = current
            current = 0
    if current:
        runs.append(current)
        if current > longest:
            longest = current
    return len(runs), longest, transitions, runs


def weighted_position_score(mask: Sequence[bool], start: int = 1) -> float:
    total = 0.0
    for idx, value in enumerate(mask, start=start):
        if value:
            total += float(idx)
    return total


def probability_to_risk_level(probability: float) -> str:
    if probability > 0.9:
        return "high"
    if probability >= 0.5:
        return "medium"
    return "low"


def risk_level_from_probability(probability: float) -> str:
    return probability_to_risk_level(probability)
