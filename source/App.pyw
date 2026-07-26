import multiprocessing

from Startup import run_standalone_mode


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_standalone_mode()
