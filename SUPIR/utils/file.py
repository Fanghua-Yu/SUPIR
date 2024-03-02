import os
from typing import List, Tuple

from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir


def load_file_list(file_list_path: str) -> List[str]:
    files = []
    # each line in file list contains a path of an image
    with open(file_list_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append(path)
    return files


def list_image_files(
    img_dir: str,
    exts: Tuple[str]=(".jpg", ".png", ".jpeg"),
    follow_links: bool=False,
    log_progress: bool=False,
    log_every_n_files: int=10000,
    max_size: int=-1
) -> List[str]:
    files = []
    for dir_path, _, file_names in os.walk(img_dir, followlinks=follow_links):
        early_stop = False
        for file_name in file_names:
            if os.path.splitext(file_name)[1].lower() in exts:
                if max_size >= 0 and len(files) >= max_size:
                    early_stop = True
                    break
                files.append(os.path.join(dir_path, file_name))
                if log_progress and len(files) % log_every_n_files == 0:
                    print(f"find {len(files)} images in {img_dir}")
        if early_stop:
            break
    return files


def get_file_name_parts(file_path: str) -> Tuple[str, str, str]:
    parent_path, file_name = os.path.split(file_path)
    stem, ext = os.path.splitext(file_name)
    return parent_path, stem, ext


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/download_util.py/
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file