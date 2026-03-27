
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_logged_command(cmd: Sequence[str], log_file: str | Path | None = None) -> None:
    if log_file is None:
        subprocess.run(list(cmd), check=True)
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write('

' + '=' * 40 + '
')
        f.write('RUN: ' + ' '.join(map(str, cmd)) + '
')
        f.write('=' * 40 + '
')
        p = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end='')
            f.write(line)
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(map(str, cmd))}")
