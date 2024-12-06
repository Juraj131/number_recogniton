import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = "cuda"
print(f'Using device: {device}')


#############################################################################################################
############################################# Transformace a načtení dat ####################################
#############################################################################################################


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Zmena veľkosti obrázkov na 28x28
    transforms.ToTensor(),        # Konverzia obrázkov na tenzory
    transforms.Normalize((0.5,), (0.5,)),  # Normalizácia obrázkov
    ])  # Binarizácia: pixely > 0.5 → 1, inak → 0

    
# Vytvorenie datasetu pomocou ImageFolder
train_dataset = datasets.ImageFolder(
    root=r'C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\final_train_data',
    transform=transform)

# Vytvorenie DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.ImageFolder(
    root=r'C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\final_val_data',
    transform=transform)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

#############################################################################################################
############################################# Architektura modelu ############################################
#################################################################################################################

class MLP_net(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size):
        super(MLP_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_3_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Example usage
input_size = 28 * 28  # Assuming input images are 28x28
hidden_1_size = 512
hidden_2_size = 256
hidden_3_size = 128
output_size = 10  # Assuming 10 classes

mymodel = MLP_net(input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size).to(device)

# Definovanie loss funkcie a optimizéra
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.001, weight_decay=1e-4)  # Pridanie weight_decay// , weight_decay=1e-5

#############################################################################################################
############################################# Funkce ########################################################
#############################################################################################################

# Funkcia na výpočet presnosti
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

#############################################################################################################
############################################# Trénování modelu ##############################################
#############################################################################################################
loss_values = []
val_loss_values = []
num_epochs = 100

for epoch in range(num_epochs):
    mymodel.train()
    start_time = time.time()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # Predpokladáme, že obrázky sú 28x28 a musia byť flattenované na 784 prvkov
        images = images.view(images.size(0), -1)
    
        # Forward pass
        outputs = mymodel(images)
        loss = loss_function(outputs, labels)
       # print(outputs)
        # Backward pass a optimalizácia
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        running_loss += loss.item()
    
    end_time = time.time()
    epoch_duration = end_time - start_time

    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Duration: {epoch_duration:.2f} seconds')
    
    #přidání výpočtu loss pro val
    mymodel.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = mymodel(images)
            loss = loss_function(outputs, labels)
            val_running_loss += loss.item()
    
    avg_val_loss = val_running_loss / len(val_loader)
    val_loss_values.append(avg_val_loss)
    #konec

    if (epoch + 1) % 10 == 0:
        accuracy, preds, labels = calculate_accuracy(train_loader, mymodel)
        print(f'Accuracy after {epoch+1} epochs: {accuracy:.2f}%')

# Celková presnosť po trénovaní
final_accuracy, train_preds, train_labels = calculate_accuracy(train_loader, mymodel)
print(f'Final Accuracy: {final_accuracy:.2f}%')

#############################################################################################################
############################################# Validace modelu ###############################################
#############################################################################################################


mymodel.eval()
val_accuracy, val_preds, vals_labels = calculate_accuracy(val_loader, mymodel)
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Vykreslení hodnot ztrátové funkce
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

 # Vytvoření a vykreslení confusion matrix pro trénovací data
train_cm = confusion_matrix(train_labels, train_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=train_cm, display_labels=train_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Training Data')
plt.show()


# Vytvoření a vykreslení confusion matrix pro validační data
val_cm = confusion_matrix(vals_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=val_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Validation Data')
plt.show()

torch.save(mymodel.state_dict(), r'C:\\Users\\juraj\\OneDrive\\Documents\\UNI_BTB\\5.semester\\UIM\\final_projekt2\\number_recogniton\\saved_weights\\mymodel.pth')
print("Model weights have been saved in the specified folder!")