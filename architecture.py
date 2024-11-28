import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


#############################################################################################################
############################################# Transformace a načtení dat ####################################
#############################################################################################################

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Zmena veľkosti obrázkov na 28x28
    transforms.ToTensor(),        # Konverzia obrázkov na tenzory
    transforms.Normalize((0.5,), (0.5,)),  # Normalizácia obrázkov
    transforms.Grayscale(num_output_channels=1)])  # Prevod na čiernobiely obrázok

# Vytvorenie datasetu pomocou ImageFolder
train_dataset = datasets.ImageFolder(
    root=r'C:\Users\USER\Desktop\muj_tretak\UIM\OCR\final_train_data',
    transform=transform)

# Vytvorenie DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#############################################################################################################
############################################# Architektura modelu ############################################
#############################################################################################################

class MLP_net(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLP_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
      #  self.dropout1 = nn.Dropout(0.3)  # Dropout s pravdepodobnosťou 0.5
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
      #  self.dropout2 = nn.Dropout(0.3)  # Dropout s pravdepodobnosťou 0.5
        self.fc3 = nn.Linear(hidden_2_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
    #    x = self.dropout1(x)  # Aplikácia dropout po prvej skrytej vrstve
        x = self.fc2(x)
        x = self.relu(x)
     #   x = self.dropout2(x)  # Aplikácia dropout po druhej skrytej vrstve
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# definice modelu
input_size = 28 * 28 # 3 farby, 28x28 veľkosť obrázka
hidden_1_size = 256
hidden_2_size = 128
output_size = 10

mymodel = MLP_net(input_size, hidden_1_size, hidden_2_size, output_size).to(device)

# Definovanie loss funkcie a optimizéra
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.001)  # Pridanie weight_decay// , weight_decay=1e-5

#############################################################################################################
############################################# Funkce ########################################################
#############################################################################################################

# Funkcia na výpočet presnosti
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

#############################################################################################################
############################################# Trénování modelu ##############################################
#############################################################################################################

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
        
        # Backward pass a optimalizácia
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Duration: {epoch_duration:.2f} seconds')
    
    if (epoch + 1) % 10 == 0:
        accuracy = calculate_accuracy(train_loader, mymodel)
        print(f'Accuracy after {epoch+1} epochs: {accuracy:.2f}%')

# Celková presnosť po trénovaní
final_accuracy = calculate_accuracy(train_loader, mymodel)
print(f'Final Accuracy: {final_accuracy:.2f}%')

#############################################################################################################
############################################# Validace modelu ###############################################
#############################################################################################################

val_dataset = datasets.ImageFolder(
    root=r'C:\Users\USER\Desktop\muj_tretak\UIM\OCR\final_val_data',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

mymodel.eval()
val_accuracy = calculate_accuracy(val_loader, mymodel)
print(f'Validation Accuracy: {val_accuracy:.2f}%')