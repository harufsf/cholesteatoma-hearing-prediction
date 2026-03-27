
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CaseInfo:
    pid: str
    dicom_dir: str
    side_rl: Optional[str]
    fold: int
    point_json: str
