import requests
from pathlib import Path

folder = Path("data/mars")
folder.mkdir(parents=True, exist_ok=True)

base_url = (
    "https://pds-geosciences.wustl.edu/"
    "mgs/mgs-m-mola-5-megdr-l3-v1/"
    "mgsl_300x/meg016/"
)

files = [
    "megt90n000eb.img",
    "megt90n000eb.lbl",
]

for filename in files:
    url = base_url + filename
    destination = folder / filename

    print(f"\nЗавантаження: {filename}")
    response = requests.get(url, timeout=120)

    print("HTTP:", response.status_code)

    if response.status_code == 200:
        destination.write_bytes(response.content)
        print(f"Збережено: {destination}")
    else:
        print("Не вдалося завантажити файл.")