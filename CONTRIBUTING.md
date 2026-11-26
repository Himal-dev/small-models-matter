# Contributing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check src scripts tests
python -m pytest -q
python scripts/benchmark.py --runs 2000
python scripts/build_pages.py
```

For dashboard changes, check the local watcher and the static replay at narrow
and wide viewports. For model changes, run `python scripts/evaluate.py` and
report which checkpoint you evaluated. Keep generated files in `dist/` and
local training outputs in `data/run/`.

For bug reports, include the command, runtime versions, and observed behavior.
For benchmark submissions, attach `python scripts/benchmark.py --json` output.
