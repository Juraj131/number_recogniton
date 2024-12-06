import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# # Label names
label_nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

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

print(sum(count_each_number(r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\train_dir")))

#########################################################################################################
############################################# Vyobrazení mé geniality #######################################
#############################################################################################################

figure_2 = plt.figure(figsize=(10, 7))

plt.bar(list(range(10)), count_each_number(r"C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\train_dir"), edgecolor='black')

# Add labels and title
plt.xlabel('Čísla')
plt.ylabel('Zastoupení')
plt.title('Visulizace zastoupení čísel v datasetu uuuuuu')

# Show the plot
plt.xticks(list(range(10)))
plt.show()
