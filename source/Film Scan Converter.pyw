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
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--directory', help='load this directory on open')
    p.add_argument('-o', '--output_directory', help='set an output directory')
    p.add_argument('-f', '--file', help='open one or more files on launch')
    return p.parse_args()


logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s:::%(levelname)s:::%(message)s'
logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format=FORMAT)
opts = getopts()
   
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

    window = GUI(root, opts.output_directory)

    # If a directory argument has been passed then load it after the mainloop starts
    if opts.directory is not None:
        root.after(0, window.resize_UI)
        root.after(0, window.load_all_from_path, opts.directory)

    # If a file argument has been passed and a directory argument has not then load
    # them after mainloop starts
    if (opts.file is not None) and (opts.directory is None):
        filenames = opts.file.split(',')        
        filenames[:] = [name.strip() for name in filenames if os.path.isfile(name.strip())]
        root.after(0, window.resize_UI)
        root.after(0, window.import_from_filenames, filenames)

    root.mainloop()
