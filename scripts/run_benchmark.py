"""
Single-command benchmark evaluator.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)
    sys.exit(subprocess.call([sys.executable, "main.py", "--benchmark"]))
