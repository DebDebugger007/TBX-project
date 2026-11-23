"""Put ``src`` on sys.path so the scripts run without installing the package.

Importing this at the top of each script lets you run them straight from a clone
(``python scripts/01_prepare_data.py``) during development. For a real install,
``pip install -e .`` makes this unnecessary.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
