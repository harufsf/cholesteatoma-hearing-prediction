from __future__ import annotations

from typing import Any, Optional

def normalize_pid(pid: Any) -> Optional[str]:
    if pid is None:
        return None
    s = str(pid).strip()
    s = s.replace("\u200b", "").replace("\ufeff", "")
    if s.endswith(".0"):
        s = s[:-2]
    try:
        s = str(int(float(s)))
    except Exception:
        pass
    return s
