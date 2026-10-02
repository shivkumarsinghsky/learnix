"""Every lesson script must run end to end, so the learning path never rots."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LESSONS = sorted((ROOT / "lessons").glob("[0-9][0-9]_*.py"))


def test_lessons_are_discovered() -> None:
    assert len(LESSONS) == 5


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda p: p.name)
def test_lesson_runs(lesson: Path, tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, str(lesson)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_lesson_1_uses_its_argument() -> None:
    # The first version ignored the function argument and read a global variable.
    namespace: dict[str, object] = {}
    exec((ROOT / "lessons" / "01_prediction.py").read_text(), namespace)
    predict = namespace["predict_salary"]
    assert predict([0, 1], 10, 5) == [5, 15]  # type: ignore[operator]
