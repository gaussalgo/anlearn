from pathlib import Path
from urllib.request import urlopen


def fetch_dataset(dataset_path: str, url: str) -> None:
    path = Path(dataset_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.is_file():
        return
    with open(path, "wb") as dataset_file, urlopen(url) as url_file:
        dataset_file.write(url_file.read())
