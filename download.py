from argparse import Namespace, ArgumentParser
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

from src.paths import DATA_DIR, DATASET_ZIP


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--force", required=False, action="store_true")
    return parser.parse_args()


def run(force: bool) -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not DATASET_ZIP.is_file() or force:
        print("Downloading...")
        with urlopen("https://web.ais.dk/aisdata/aisdk-2024-05-04.zip") as response:
            total_size = int(response.headers.get("content-length", 0))
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

            with open(DATASET_ZIP, "wb") as file:
                while True:
                    buffer = response.read(1024)
                    if not buffer:
                        break

                    file.write(buffer)
                    progress_bar.update(len(buffer))

            progress_bar.close()

    print("Extracting...")
    with ZipFile(DATASET_ZIP, "r") as file:
        file.extractall(DATA_DIR)

    print("Done.")


if __name__ == "__main__":
    run(**vars(parse_args()))
