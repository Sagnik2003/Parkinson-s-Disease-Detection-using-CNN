
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
class ParkinsonCNN(nn.Module):
    def __init__(self):
        super(ParkinsonCNN, self).__init__()
        self.convo_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            
        )
        self.flatten = nn.Flatten()
        self.leaky_ffn1 = nn.Sequential(
            nn.Linear(14*14*128, 128),
            nn.LeakyReLU(),
            
        )
        self.leaky_ffn2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            
        )
        
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = x.to(device)
        x = self.convo_layers(x)
        x = self.flatten(x)
        x = self.leaky_ffn1(x)
        x = self.leaky_ffn2(x)
        x = self.output(x)
        return x

# Initialize Model
Model = ParkinsonCNN().to(device)



criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(Model.parameters(), lr=0.001, weight_decay=0.0035)

optimizer = optim.Adam([
    {'params': Model.convo_layers.parameters(), 'weight_decay': 0.0},
    {'params': Model.leaky_ffn1.parameters(), 'weight_decay': 0.0035},
    {'params': Model.leaky_ffn2.parameters(), 'weight_decay': 0.0035},
    {'params': Model.output.parameters(), 'weight_decay': 0.0}
], lr=0.001)

# print(Model.load_state_dict)