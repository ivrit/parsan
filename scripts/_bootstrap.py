"""Make `parsan` importable when running scripts directly (no install needed).
Each CLI does `import _bootstrap` first. On the HPC the repo is just synced, not pip-installed."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
