from datetime import datetime
import logging
import os


LOG_FORMAT = '%(asctime)s:::%(levelname)s:::%(message)s'


def create_log_path(launch_time=None):
    if launch_time is None:
        launch_time = datetime.now()

    local_appdata = os.environ.get(
        'LOCALAPPDATA',
        os.path.join(os.path.expanduser('~'), '.local', 'share'),
    )
    log_directory = os.path.join(
        local_appdata, 'Film Scan Converter', 'logs'
    )
    os.makedirs(log_directory, exist_ok=True)
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
