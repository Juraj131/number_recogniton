import os
import shutil

"""
    skript rozděluje čísla do složek podle jejich názvu, zároveň vyvtoří 10 složek 0-9, do kterých se čísla přesunou
"""
# Zdrojový priečinok s obrázkami
source_dir = r'C:\Users\juraj\OneDrive\Documents\UNI_BTB\5.semester\UIM\final_projekt2\number_recogniton\val_dir'

# Cieľový priečinok, kde budú obrázky roztriedené do priečinkov 0 až 9
target_dir = r'C:\Users\juraj\OneDrive\Documents\UNI_BTB\5.semester\UIM\final_projekt2\number_recogniton\final_val_data' # pre vytvorenie viac priecinkov staci ymenit adresu

# Mapovanie anglických názvov číslic na čísla
digit_names = {
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9'
}

# Vytvor cieľové priečinky, ak neexistujú
for i in range(10):
    os.makedirs(os.path.join(target_dir, str(i)), exist_ok=True)

# Prejdi všetky súbory v zdrojovom priečinku
for filename in os.listdir(source_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Skontroluj, či je súbor obrázok
        # Rozpoznaj anglický názov číslice v názve súboru
        for digit_name, digit_str in digit_names.items():
            if digit_name in filename.lower():
                # Cieľový priečinok pre danú číslicu
                target_subdir = os.path.join(target_dir, digit_str)
                # Cesta k zdrojovému súboru
                source_file = os.path.join(source_dir, filename)
                # Cesta k cieľovému súboru
                target_file = os.path.join(target_subdir, filename)
                # Presuň súbor do cieľového priečinka
                shutil.copy(source_file, target_file)
                break

print('Obrázky boli úspešne roztriedené.')