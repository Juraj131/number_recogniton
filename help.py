import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# # Label names
label_nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Function to load images from folder and return images and their labels
# def load_images_from_folder(folder, image_size=(28, 28)):
#     images = []
#     labels = []
    
#     for filename in os.listdir(folder):  # v os.listdir(folder) jsou názvy souborů v daném adresáři
#         img_path = os.path.join(folder, filename) # v img_path je cesta k danému souboru (např. "train_dir\\three_001.png")
        
#         # Find the label from the folder name based on the image filename
#         for label_num in label_nums:
#             if label_num in img_path:
#                 label = label_nums.index(label_num)
#                 break  # Once we find the label, no need to check further
        
#         try:
#             # Read image using OpenCV (grayscale)
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 raise ValueError(f"Image at {img_path} is invalid.")  # Handle case where the image doesn't load properly
            
#             img_mirrored = mirrored(img)
#             img_rotated = rotated(img)
#             img_noisy = noisy(img)
#             # Normalize the image pixels by dividing by 255 <- nevím jestli tohle nebude dělat bordel potom u šumu
#             img = img / 255.0  # Pixel values in the range [0, 1]

#             img = cv2.resize(img, image_size)  # Resize image to 28x28
#             img = img.flatten()  # Flatten the image to a 1D vector
            
#             images.append(img)
#             labels.append(label)  # Assign the label
#         except Exception as e:
#             print(f"Error loading image {filename}: {e}")

#     return np.array(images), np.array(labels)

# # Loading the datasets
# X_train, y_train = load_images_from_folder("train_dir")
# X_val, y_val = load_images_from_folder("val_dir")
# X_test, y_test = load_images_from_folder("test_dir")

# # Convert labels to one-hot encoding
# def one_hot_encode(labels, num_classes=10):
#     return np.eye(num_classes)[labels]

# # One-hot encode the labels
# y_train = one_hot_encode(y_train)
# y_val = one_hot_encode(y_val)
# y_test = one_hot_encode(y_test)

# # Check the shape of the data
# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_val shape: {X_val.shape}")
# print(f"y_val shape: {y_val.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_test shape: {y_test.shape}")

# print(y_train[0])  # Example one-hot encoded label

#############################################################################################################
################################# Počítání zastoupení jednotlivých čísel v datasetu #########################
#############################################################################################################

def count_each_number(folder):
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

print(count_each_number("train_dir"))  # [250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
# průměrný počet jednoho čísla je 250.2

#############################################################################################################
############################################# Definování funkcí #############################################
#############################################################################################################

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

#############################################################################################################

img = cv2.imread(f"C:\\Users\\USER\\Desktop\\muj_tretak\\UIM\\OCR\\five_128.png")

img_noisy = noisy(img)
img_mirrored = mirrored(img)
img_rotated = rotated(img)

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
