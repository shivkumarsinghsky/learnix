# Contributing to learnix

Thanks for your interest in `learnix`. Corrections to the maths, clearer explanations and new
lessons are all welcome.

## Workflow

1. Fork and clone, then install: `pip install -e ".[dev]"`.
2. Open an issue before starting a new lesson, so topics follow the roadmap order.
3. Before opening a pull request, run:
   ```bash
   pytest
   ruff check . && ruff format --check .
   mypy
   ```

## Adding a lesson

1. Add `lessons/NN_topic.py`. Keep it short, runnable on its own and explained in its docstring.
2. If the concept belongs in the package, add it under `src/learnix/` with type hints.
3. Add tests that compare against an independent reference (an exact formula, numerical
   derivatives or scikit-learn), not only against printed output. See ADR-003.
4. Add the lesson to the learning-path table in the README, and update `tests/test_lessons.py`'s
   expected lesson count.

## Style

Lesson files may use simple, explicit loops even where a comprehension would be shorter;
readability for a beginner comes first. Package code follows Ruff and strict mypy.
