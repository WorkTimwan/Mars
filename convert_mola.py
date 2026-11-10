import rasterio

# Шлях до файлу опису MOLA
lbl_path = "data/mars/megt90n000eb.lbl"
tif_path = "data/mars/mars_dem.tif"

print("Відкриваємо марсіанські дані...")
with rasterio.open(lbl_path) as src:
    print(f"Розміри карти: {src.width} на {src.height} пікселів")
    data = src.read(1)
    
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff"})
    
    print("Зберігаємо у формат GeoTIFF...")
    with rasterio.open(tif_path, "w", **out_meta) as dest:
        dest.write(data, 1)

print("Готово! Файл mars_dem.tif успішно створено.")