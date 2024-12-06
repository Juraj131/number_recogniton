import shutil
from sklearn.model_selection import train_test_split
import os
"""
    rozdělí data (obrázky) do tří složek: train_dir, val_dir, test_dir s poměrem 80:10:10,
    data bere ze source_dir, !cesty ke sloužkám se musí změnit! 
"""
# Tento file splitne obrazky do třech složek: train_dir, val_dir, test_dir s poměrem 80:10:10

source_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\trainData"
train_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\train_dir"
val_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\val_dir"
test_dir = r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\test_dir"

# Split the data into training, validation, and test sets

# List all images
all_images = os.listdir(source_dir)

# First, split into 80% train and 20% remaining (for validation + test)
train_images, val_test_images = train_test_split(all_images, test_size=0.2, random_state=42)

# Then, split the remaining 20% into 10% validation and 10% test
val_images, test_images = train_test_split(val_test_images, test_size=0.5, random_state=42)

# Function to copy images to their respective directories
def copy_images(image_list, destination_dir):
    for image in image_list:
        shutil.copy(os.path.join(source_dir, image), os.path.join(destination_dir, image))

# Copy images
copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)



