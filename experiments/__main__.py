"""``python -m experiments`` entry point.

Forwards every argument to :func:`scripts.run_experiment.main` so the
single-experiment CLI is invocable as a module without putting
``scripts/`` on ``PATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is sibling to experiments/, not a package — add the repo
# root to sys.path so the import resolves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_experiment import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
