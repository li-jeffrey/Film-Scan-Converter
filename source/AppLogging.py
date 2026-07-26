from datetime import datetime
import logging
import os

from AppPaths import ensure_app_data_directory


LOG_FORMAT = '%(asctime)s:::%(levelname)s:::%(message)s'


def create_log_path(launch_time=None):
    if launch_time is None:
        launch_time = datetime.now()

    log_directory = ensure_app_data_directory('logs')
    timestamp = launch_time.strftime('%Y%m%d-%H%M%S')
    return os.path.join(
        log_directory, f'film-scan-converter-{timestamp}.log'
    )


def configure_logging(launch_time=None):
    logging.basicConfig(
        filename=create_log_path(launch_time),
        level=logging.DEBUG,
        format=LOG_FORMAT,
    )
