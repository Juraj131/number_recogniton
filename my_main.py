import os
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

label_nums = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def load_images_from_folder(folder, image_size=(28, 28)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        for label_num in label_nums:
            if label_num in img_path:
                label = label_nums.index(label_num)
        try:
            # Read image using OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv2.resize(img, image_size)  # Resize image to 28x28
            img = img.flatten()  # Flatten the image to a 1D vector
            
            # Normalize the image pixels by dividing by 255
            img = img / 255.0  # Now pixel values will be in the range [0, 1]
            
            images.append(img)
            labels.append(label)  # Assign the label
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images, labels

X_train, y_train = load_images_from_folder("train_dir")
X_val, y_val = load_images_from_folder("val_dir")
X_test, y_test = load_images_from_folder("test_dir")

# # Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Step 2: Preprocessing - Scale the data (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale the training data
X_val_scaled = scaler.transform(X_val)          # Use the same scaler for validation data
X_test_scaled = scaler.transform(X_test)        # Use the same scaler for test data

# Step 3: Define the MLP Model
# You can adjust the architecture (e.g., number of hidden layers, neurons per layer)
mlp = MLPClassifier(hidden_layer_sizes=(128,),  # One hidden layer with 128 neurons
                    activation='relu',          # Use ReLU activation function
                    solver='adam',              # Adam optimizer
                    max_iter=300,               # Maximum number of iterations
                    random_state=42)            # Set random state for reproducibility

# Step 4: Train the MLP Model
mlp.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
# Predict on the test set
y_pred = mlp.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')