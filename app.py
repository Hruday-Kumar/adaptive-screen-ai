import os
import sys

BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ui.dashboard import run_dashboard  # noqa: E402


if __name__ == "__main__":
    run_dashboard()
