import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the image
image = cv2.imread('eight_001.png')

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to 28x28 pixels
image_resized = cv2.resize(image, (28, 28))

# Convert the resized image to grayscale
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)

# Normalize the image
image_matrix = np.array(image_gray)

image_matrix = image_matrix / 255.0

# Display the image   vykreslení image_matrix a image_gray/255.0 je to samé (asi)
# plt.imshow(image_matrix, cmap='gray')
# plt.axis('off')  # Hide the axis
# plt.show()

# Flatten the image
image_matrix = image_matrix.flatten()


# cv2.imshow("Grayscale Image", image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the final preprocessed image in the directory "processed_data"
# output_dir = 'processed_data'
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, 'eight_001_processed.png')
# cv2.imwrite(output_path, image_gray)