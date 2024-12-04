import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import shutil
from PIL import Image, ImageOps
source_dir = r"C:\Users\USER\Desktop\muj_tretak\UIM\OCR\train_dir"

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
'''
def noisy(image):
    row,col,ch= image.shape
    mean = random.uniform(-20, 20,)
    var = random.uniform(0.01, 30)
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def mirrored(image):
    return image[:, ::-1, :]

def rotated(image):
    (h, w) = image.shape[:2]    
    # Calculate the center of the image
    center = (w / 2, h / 2)

    # Rotation angle in degrees
    angle = random.randint(-40, 40)
    # Scale
    scale = 1.0

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Rotate the image
    img_rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img_rotated
'''

def noisy(image):
    np_image = np.array(image)
    row, col, ch = np_image.shape
    mean = random.uniform(-20, 20)
    var = random.uniform(10, 50)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np_image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

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

#tato funkce uděla z každého obrázku 3 nové obrázky, jedno zrcadlení, jedno rotaci a jedno s šumem
augment_images()

