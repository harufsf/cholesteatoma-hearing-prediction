from __future__ import annotations

from typing import Iterable


def build_category_map(values: Iterable[object]) -> dict[str, int]:
    uniq = sorted({str(v) for v in values if str(v) != "nan"})
    out = {"__UNK__": 0}
    for i, v in enumerate(uniq, start=1):
        out[v] = i
    return out


def map_category(value: object, mapping: dict[str, int]) -> int:
    return mapping.get(str(value), mapping.get("__UNK__", 0))
