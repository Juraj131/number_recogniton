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

import os
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

label_nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Function to load images from folder and return images and their labels
def load_images_from_folder(folder, image_size=(28, 28)):
    images = []
    labels = []
    
    for filename in os.listdir(folder):  # v os.listdir(folder) jsou názvy souborů v daném adresáři
        img_path = os.path.join(folder, filename) # v img_path je cesta k danému souboru (např. "train_dir\\three_001.png")
        
        # Find the label from the folder name based on the image filename
        for label_num in label_nums:
            if label_num in img_path:
                label = label_nums.index(label_num)
                break  # Once we find the label, no need to check further
        
        try:
            # Read image using OpenCV (grayscale)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Image at {img_path} is invalid.")  # Handle case where the image doesn't load properly
            
            # Normalize the image pixels by dividing by 255 <- nevím jestli tohle nebude dělat bordel potom u šumu
            img = img / 255.0  # Pixel values in the range [0, 1]

            img = cv2.resize(img, image_size)  # Resize image to 28x28
            img = img.flatten()  # Flatten the image to a 1D vector
            
            images.append(img)
            labels.append(label)  # Assign the label
        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    return np.array(images), np.array(labels)