import os

import requests, re


def download_file(url, path):
    try:
        print(f'Attempting to download file \'{get_path_filename(path)}\'...')
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f'Successfully downloaded file \'{get_path_filename(path)}\'')
    except requests.exceptions.RequestException as e:
        print(f'Error downloading file at \'{url}\'\n{e}')


def get_path_filename(path):
    try:
        return re.search(r'[^/]+$', path).group(0)
    except AttributeError:
        return ''


def check_files_exist(*paths):
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
