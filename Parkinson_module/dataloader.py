from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import numpy as np


data_dir = 'Geometric_Augmentation/training_set/spiral' # Directory containing the training images

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)
# Split dataset into training and validation sets
def create_dataloaders(dataset, split_ratio=0.8, batch_size=32):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(split_ratio * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_indices, val_indices


train_loader, val_loader, train_indices, val_indices = create_dataloaders(dataset)

# print(len(train_loader.dataset))
# print(len(train_loader.dataset), len(val_loader.dataset), len(train_indices), len(val_indices))