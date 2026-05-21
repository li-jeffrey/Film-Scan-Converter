import tkinter as tk
import multiprocessing
import ctypes
import os
import sys
import logging
import argparse

#Custom classes
from GUI import GUI


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

        def resource_path(relative_path):    
            try:       
                base_path = sys._MEIPASS
                return os.path.join(base_path, icon)
            except Exception:
                base_path = os.path.abspath('.')
                return os.path.join(base_path, relative_path)

        icon = 'camera-roll.ico'
        if not hasattr(sys, 'frozen'):
            datafile = os.path.join(os.path.dirname(__file__), 'assets', icon)
        else:
            datafile = os.path.join(sys.prefix, 'assets', icon)
    except Exception as e:
        logger.exception(f'Exception: {e}')
        root = tk.Tk()
    else:
        root = tk.Tk()
        root.iconbitmap(default=resource_path(datafile))

    window = GUI(root, lightroom_path=lightroom_path)

    if lightroom_path is not None:
        root.after(0, window.resize_UI)
        root.after(0, window.import_lightroom_edit_in, lightroom_path)

    root.mainloop()
