"""Convenience launcher for GestureFlow."""

import subprocess
import sys
import os

if __name__ == "__main__":
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    subprocess.run([sys.executable, os.path.join(src_dir, "main.py")], cwd=src_dir)
