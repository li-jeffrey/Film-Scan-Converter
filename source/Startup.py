import tkinter as tk
import ctypes
import os
import sys
import logging
import argparse
import queue
import socket
import threading

from AppLogging import configure_logging


configure_logging()

# Custom classes
from GUI import GUI
from LightroomEditInUI import LightroomEditInUI
from ResourcePaths import resource_path


IPC_HOST = '127.0.0.1'
IPC_PORT = 49152

logger = logging.getLogger(__name__)


def getopts():
    p = argparse.ArgumentParser(
        description='Film Scan Converter'
    )
    p.add_argument(
        'image_paths',
        nargs='*',
        help='image file(s) to open',
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


def create_root(start_hidden=False):
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        datafile = os.path.join('assets', 'camera-roll.ico')
    except Exception as exception:
        logger.exception(f'Exception: {exception}')
        root = tk.Tk()
    else:
        root = tk.Tk()
        root.iconbitmap(default=resource_path(datafile))

    if start_hidden:
        root.withdraw()
    return root


def create_lightroom_listener():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((IPC_HOST, IPC_PORT))
        server.listen(1)
    except OSError:
        server.close()
        return None
    return server


def listen_for_lightroom_paths(server, received_paths):
    def listen():
        while True:
            try:
                connection, _ = server.accept()
                with connection:
                    chunks = []
                    while True:
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                path = b''.join(chunks).decode('utf-8').strip()
                if path:
                    received_paths.put(path)
            except (OSError, UnicodeError):
                break

    threading.Thread(target=listen, daemon=True).start()


def send_path_to_lightroom_instance(path):
    try:
        with socket.create_connection((IPC_HOST, IPC_PORT), timeout=2) as client:
            client.sendall(path.encode('utf-8'))
    except OSError as exception:
        logger.exception(f'Could not hand image to Lightroom instance: {exception}')
        return False
    return True


def run_lightroom_mode():
    opts = getopts()
    lightroom_path = resolve_lightroom_path(opts.image_paths[0] if opts.image_paths else None)
    server = create_lightroom_listener()

    if server is None:
        if lightroom_path is not None:
            send_path_to_lightroom_instance(lightroom_path)
        return

    received_paths = queue.Queue()
    listen_for_lightroom_paths(server, received_paths)
    root = create_root(start_hidden=lightroom_path is None)
    window = None

    def open_image(path):
        nonlocal window
        path = resolve_lightroom_path(path)
        if path is None:
            return

        if window is None:
            window = LightroomEditInUI(root, path)
            root.deiconify()
        root.lift()
        root.focus_force()
        root.after(0, window.resize_UI)
        root.after(0, window.import_lightroom_edit_in, path)

    def process_received_paths():
        try:
            while True:
                open_image(received_paths.get_nowait())
        except queue.Empty:
            pass
        root.after(100, process_received_paths)

    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.after(100, process_received_paths)
    if lightroom_path is not None:
        open_image(lightroom_path)

    try:
        root.mainloop()
    finally:
        server.close()


def run_standalone_mode():
    opts = getopts()
    image_paths = opts.image_paths
    paths = tuple(os.path.abspath(path.strip().strip('"')) for path in image_paths)
    missing_paths = [path for path in paths if not os.path.isfile(path)]
    if missing_paths:
        print(f'Error: file not found: {missing_paths[0]}', file=sys.stderr)
        return

    root = create_root()
    window = GUI(root)
    if paths:
        root.after(0, window.resize_UI)
        root.after(0, window.import_from_filenames, paths)
    root.mainloop()
