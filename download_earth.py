
import requests
from pathlib import Path

# Тестова ділянка Землі: 40° пн. ш., 0° сх. д.
# Copernicus DEM GLO-90

name = "Copernicus_DSM_COG_30_N40_00_E000_00_DEM"

url = (
    "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/"
    + name + "/" + name + ".tif"
)

folder = Path("data/earth")
folder.mkdir(parents=True, exist_ok=True)

destination = folder / "earth_test.tif"

print("Перевіряємо доступ до датасету...")

try:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Завантажено: {downloaded / 1024 / 1024:.1f} MB")

    print("\nУСПІХ!")
    print("Файл збережено:", destination.resolve())

except Exception as e:
    print("\nПомилка завантаження:", e)
