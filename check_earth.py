
import rasterio
import numpy as np

path = "data/earth/earth_test.tif"

with rasterio.open(path) as src:
    data = src.read(1, masked=True)

    print("\n===== ДАНІ ЗЕМЛІ =====")
    print("Розмір:", src.width, "x", src.height)
    print("Система координат:", src.crs)
    print("Роздільність:", src.res)
    print("Межі:", src.bounds)
    print("Мінімальна висота:", data.min(), "м")
    print("Максимальна висота:", data.max(), "м")
    print("Середня висота:", data.mean(), "м")

print("\nПеревірку завершено!")
