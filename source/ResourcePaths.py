import os
import sys


def resource_path(relative_path):
    """Return an asset path for source and PyInstaller executions."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)
