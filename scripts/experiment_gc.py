"""Run experiment entropy management checks.

Usage:
    python scripts/experiment_gc.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness_control import EntropyManager


def main():
    manager = EntropyManager(project_root=ROOT)
    path = manager.save_report()
    print(f"Entropy management report saved to: {path}")


if __name__ == "__main__":
    main()
