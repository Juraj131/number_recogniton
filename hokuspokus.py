import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import time

# Kontrola dostupnosti GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Definovanie transformácií pre obrázky
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Zmena veľkosti obrázkov na 28x28
    transforms.Grayscale(num_output_channels=1),  # Prevod na čiernobiely obrázok
    transforms.ToTensor(),  # Konverzia obrázkov na tenzory
    transforms.Normalize((0.5,), (0.5,))  # Normalizácia obrázkov
])

# Definovanie MLP architektúry
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image to a 1D vector
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

# Funkcia na výpočet presnosti
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    # Vytvorenie datasetu pomocou ImageFolder
    train_dataset = datasets.ImageFolder(
        root=r'C:\Users\USER\Desktop\muj_tretak\UIM\OCR\final_train_data',
        transform=transform
    )

    # Vytvorenie DataLoader
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Hyperparameters
    input_size = 28 * 28  # 28x28 images flattened
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10  # 10 classes for digits 0-9
    learning_rate = 0.001
    num_epochs = 50

    # Initialize the model, loss function, and optimizer
    mymodel = MLP(input_size, hidden_size1, hidden_size2, output_size).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(mymodel.parameters(), lr=learning_rate)

    # Trénovacia slučka
    for epoch in range(num_epochs):  # Počet epoch
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

    # Vyhodnocení modelu
    final_accuracy = calculate_accuracy(train_loader, mymodel)
    print(f'Final Accuracy: {final_accuracy:.2f}%')

    val_dataset = datasets.ImageFolder(
        root=r'C:\Users\USER\Desktop\muj_tretak\UIM\OCR\final_val_data',
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    val_acc = calculate_accuracy(val_loader, mymodel)
    print(f'Validation Accuracy: {val_acc:.2f}%')