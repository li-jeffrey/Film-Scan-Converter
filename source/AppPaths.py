import os


def app_data_path(*parts):
    local_appdata = os.environ.get(
        'LOCALAPPDATA',
        os.path.join(os.path.expanduser('~'), '.local', 'share'),
    )
    return os.path.join(
        local_appdata,
        'Film Scan Converter',
        *parts,
    )


def ensure_app_data_directory(*parts):
    directory = app_data_path(*parts)
    os.makedirs(directory, exist_ok=True)
    return directory
