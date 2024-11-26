# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:46:32 2022

@author: rredi
"""

def DataPreprocessing(inputData):
    """
    Funkce slouzi pro predzpracovani dat, ktera slouzi k testovani modelu. Veskery kod, ktery vedl k nastaveni
    jednotlivych kroku predzpracovani (vcetne vypoctu konstant, prumeru, smerodatnych odchylek, atp.) budou odevzdany
    spolu s celym projektem.

    :parameter inputData:
        Vstupni data, ktera se budou predzpracovavat.
    :return preprocessedData:
        Predzpracovana data na vystup
    """
    preprocessedData = 0

    return preprocessedData

#############################################################################################################
################################# Počítání zastoupení jednotlivých čísel v datasetu #########################
#############################################################################################################
import os 
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

label_nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def count_each_number(folder):
    '''Funkce, která spočítá, kolikrát se v daném adresáři vyskytuje každé číslo'''
    number_count = []
    for number_idx in range(10):
        count = 0
        current_number = label_nums[number_idx]
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if current_number in img_path:
                count += 1
        number_count.append(count)
    return(number_count)

print(min(count_each_number("train_dir")))
# průměrný počet jednoho čísla je 250.2

#############################################################################################################
############################################# Definování funkcí #############################################
#############################################################################################################

def noisy(image):
    '''Funkce, která přidá šum do obrázku'''
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
    '''Funkce, která obrázek zrcadlově překlopí'''
    return image[:, ::-1, :]

def rotated(image):
    '''Funkce, která obrázek otočí o náhodný úhel'''
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

#############################################################################################################

img = cv2.imread(f"C:\\Users\\USER\\Desktop\\muj_tretak\\UIM\\OCR\\five_128.png")

#############################################################################################################
############################################# Vyobrazení mé geniality #######################################
#############################################################################################################

rows = 1
columns = 4

fig = plt.figure(figsize=(10, 7))

fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(img) 
plt.axis('off') 
plt.title("Original") 

# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
plt.imshow(img_rotated) 
plt.axis('off') 
plt.title("rotated")

fig.add_subplot(rows, columns, 3)

plt.imshow(img_noisy)
plt.axis('off')
plt.title("Noisy")

fig.add_subplot(rows, columns, 4)

plt.imshow(img_mirrored)
plt.axis('off')
plt.title("Mirrored")

plt.show()