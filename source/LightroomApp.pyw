import multiprocessing

from Startup import run_lightroom_mode


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_lightroom_mode()
