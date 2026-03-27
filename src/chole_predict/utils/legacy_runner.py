
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def repo_root_from_file(file: str | Path) -> Path:
    p = Path(file).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / 'src').exists() and (parent / 'scripts').exists():
            return parent
    return p.parent


def run_legacy_script(script_path: str | Path, args: Sequence[str], cwd: str | Path | None = None) -> None:
    script_path = Path(script_path).resolve()
    workdir = Path(cwd).resolve() if cwd is not None else repo_root_from_file(script_path)
    env = os.environ.copy()
    src_dir = str((repo_root_from_file(script_path) / 'src').resolve())
    env['PYTHONPATH'] = src_dir + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [sys.executable, str(script_path), *map(str, args)]
    subprocess.run(cmd, cwd=str(workdir), env=env, check=True)
