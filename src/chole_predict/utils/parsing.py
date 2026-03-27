from __future__ import annotations

from typing import List, Sequence, Tuple


def parse_csv_list(s: str | Sequence[str] | None) -> List[str]:
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s if str(x).strip()]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_csv(s: str) -> List[int]:
    return [int(float(x)) for x in parse_csv_list(s)]


def parse_shape_csv(s: str) -> Tuple[int, int, int]:
    vals = parse_int_csv(s)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 integers, got: {s}")
    return int(vals[0]), int(vals[1]), int(vals[2])
