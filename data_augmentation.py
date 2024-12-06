'''
import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import shutil
from PIL import Image, ImageOps
source_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\train_dir"

""""
    ve skriptě jsou 4 funkce, tři jsou pro augmentaci obrázků a jedna pro jejich uložení, pro uložení -> augment_images()
"""

#############################################################################################################
############################################# Definování funkcí #############################################
#############################################################################################################

def augment_images():
    image_list = os.listdir(source_dir)
    for image_name in image_list:

        image_path = os.path.join(source_dir, image_name)
        image = Image.open(image_path)

        rotated_image = rotated(image)
        rotated_image.save(os.path.join(source_dir, f"rotated_{image_name}"))

        mirrored_image = mirrored(image)
        mirrored_image.save(os.path.join(source_dir, f"mirrored_{image_name}"))

        noisy_image = noisy(image)
        noisy_image.save(os.path.join(source_dir, f"noisy_{image_name}"))


def noisy(image):
    np_image = np.array(image)
    row, col = np_image.shape[:2]
    ch = 1 if len(np_image.shape) == 2 else np_image.shape[2]
    mean = random.uniform(-20, 20)
    var = random.uniform(10, 50)
    sigma = var ** 0.5
    if ch == 1:
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss[:, :, np.newaxis]  # Pridajte nový rozmer pre grayscale obrázky
    else:
        gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np_image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy.squeeze())  # Odstráňte nadbytočný rozmer pre grayscale obrázky

def mirrored(image):
    return ImageOps.mirror(image)


def rotated(image):
    angle = random.randint(-40, 40)
    np_image = np.array(image)
    (h, w) = np_image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(rotated_image)

#tata funkce uděla z každého obrázku 3 nové obrázky, jedno zrcadlení, jedno rotaci a jedno s šumem
augment_images()
'''

import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

source_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\train_dir"

"""
    Tento skript obsahuje 4 funkcie: tri pre augmentáciu obrázkov a jednu pre ich uloženie. 
    Funkcia `augment_images()` prevedie všetky obrázky na odtiene šedej a aplikujé augmentácie.
"""

#############################################################################################################
############################################# Definovanie funkcií #############################################
#############################################################################################################

def convert_to_grayscale():
    """Prevedie všetky obrázky v zdrojovom adresári na stupne šedej."""
    image_list = os.listdir(source_dir)
    for image_name in image_list:
        image_path = os.path.join(source_dir, image_name)
        image = Image.open(image_path)
        grayscale_image = ImageOps.grayscale(image)
        grayscale_image.save(image_path)

def augment_images():
    """Aplikuj augmentácie na všetky obrázky v zdrojovom adresári."""
    convert_to_grayscale()
    image_list = os.listdir(source_dir)
    for image_name in image_list:

        image_path = os.path.join(source_dir, image_name)
        image = Image.open(image_path)

        rotated_image = rotated(image)
        rotated_image.save(os.path.join(source_dir, f"rotated_{image_name}"))

        mirrored_image = mirrored(image)
        mirrored_image.save(os.path.join(source_dir, f"mirrored_{image_name}"))

        noisy_image = noisy(image)
        noisy_image.save(os.path.join(source_dir, f"noisy_{image_name}"))

def noisy(image):
    """Pridá šum do obrázka."""
    np_image = np.array(image)
    row, col = np_image.shape
    mean = random.uniform(-20, 20)
    var = random.uniform(10, 50)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = np_image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def mirrored(image):
    """Vytvorí zrkadlený obrázok."""
    return ImageOps.mirror(image)

def rotated(image):
    """Rotuje obrázok o náhodný uhol."""
    angle = random.randint(-40, 40)
    np_image = np.array(image)
    (h, w) = np_image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(rotated_image)

# Spustenie augmentácie
augment_images()
