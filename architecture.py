import torch
import torch.nn as nn

class MLP_net(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size):
        super(MLP_net, self).__init__()
        #nn.Linear je jedna vrstva neuronky, která provádí výpočet y = x*W.T + b
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, output_size)
        self.relu = nn.ReLU() #Idk jestli stačí jedna activační funkce pro všechny vrstvy nebo by každá vrstva měla mít svoji
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
    
    loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

our_model = MLP_net(784, 128, 64, 10)
print(our_model)