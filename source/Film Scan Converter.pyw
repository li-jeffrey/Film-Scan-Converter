import tkinter as tk
import multiprocessing
import ctypes
import os
import sys
import logging
import argparse

#Custom classes
from GUI import GUI
from LightroomEditInUI import LightroomEditInUI
from ResourcePaths import resource_path


def getopts():
    p = argparse.ArgumentParser(
        description='Film Scan Converter. Pass a TIFF path when launched from Lightroom Edit In.'
    )
    p.add_argument(
        'image_path',
        nargs='?',
        help='16-bit TIFF from Lightroom Edit In (overwritten on save)',
    )
    return p.parse_args()


def resolve_lightroom_path(image_path):
    if not image_path:
        return None
    path = os.path.abspath(image_path.strip().strip('"'))
    if not os.path.isfile(path):
        print(f'Error: file not found: {path}', file=sys.stderr)
        sys.exit(1)
    if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff'):
        print(f'Error: Lightroom Edit In requires a TIFF file: {path}', file=sys.stderr)
        sys.exit(1)
    return path


logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s:::%(levelname)s:::%(message)s'
logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format=FORMAT)
opts = getopts()
lightroom_path = resolve_lightroom_path(opts.image_path)

if __name__ == '__main__':
    # Main function
    multiprocessing.freeze_support()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        icon = 'camera-roll.ico'
        datafile = os.path.join('assets', icon)
    except Exception as e:
        logger.exception(f'Exception: {e}')
        root = tk.Tk()
    else:
        root = tk.Tk()
        root.iconbitmap(default=resource_path(datafile))

    if lightroom_path is None:
        window = GUI(root)
    else:
        window = LightroomEditInUI(root, lightroom_path)
        root.after(0, window.resize_UI)
        root.after(0, window.import_lightroom_edit_in, lightroom_path)

    root.mainloop()
