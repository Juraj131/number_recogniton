# import os
# from matplotlib import pyplot as plt
# import random
# from PIL import Image, ImageOps
# import numpy as np
# import cv2

# path = r"C:\Users\USER\Desktop\muj_tretak\UIM\OCR\five_128.png"

# rows = 1
# columns = 4

# def noisy(image):
#     np_image = np.array(image)
#     row, col, ch = np_image.shape
#     mean = -20
#     var = 50
#     sigma = var ** 0.5
#     gauss = np.random.normal(mean, sigma, (row, col, ch))
#     noisy = np_image + gauss
#     noisy = np.clip(noisy, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy)

# def mirrored(image):
#     return ImageOps.mirror(image)

# def rotated(image):
#     angle = random.randint(-40, 40)
#     np_image = np.array(image)
#     (h, w) = np_image.shape[:2]
#     center = (w / 2, h / 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated_image = cv2.warpAffine(np_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
#     return Image.fromarray(rotated_image)

# img = Image.open(path)
# img_noisy = noisy(img)
# img_mirrored = mirrored(img)
# img_rotated = rotated(img)

# fig = plt.figure(figsize=(10, 7))

# fig.add_subplot(rows, columns, 1) 
  
# # showing image 
# plt.imshow(img) 
# plt.axis('off') 
# plt.title("Original") 

# # Adds a subplot at the 2nd position 
# fig.add_subplot(rows, columns, 2) 
  
# plt.imshow(img_rotated) 
# plt.axis('off') 
# plt.title("rotated")

# fig.add_subplot(rows, columns, 3)

# plt.imshow(img_noisy)
# plt.axis('off')
# plt.title("Noisy")

# fig.add_subplot(rows, columns, 4)

# plt.imshow(img_mirrored)
# plt.axis('off')
# plt.title("Mirrored")

# plt.show()
import cv2


image_cv2 = cv2.imread(r"C:\Users\USER\Desktop\muj_tretak\UIM\OCR\train_dir\five_img006-00002.png")
print(image_cv2.shape)
img = cv2.imread(r"C:\Users\USER\Desktop\muj_tretak\UIM\OCR\train_dir\five_img006-00003.png")
print(img.shape)
