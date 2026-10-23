from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
commands = [
    [sys.executable, str(ROOT / "tests" / "run_tests.py")],
    [sys.executable, str(ROOT / "code" / "run_forge_ym.py"), "--stage", "smoke"],
    [sys.executable, str(ROOT / "code" / "run_forge_ym.py"), "--stage", "structure"],
    [sys.executable, str(ROOT / "code" / "run_forge_ym.py"), "--stage", "spectrum"],
    [sys.executable, str(ROOT / "code" / "run_transfer_program.py"), "--stage", "all"],
]
for command in commands:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
