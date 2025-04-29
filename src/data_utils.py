import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_3D_dataset(data_dir="./data", batch_size=64, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Define transformations
    # Image preprocessing and data augmentation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128 pixels
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])
    
    # Load MNIST dataset as a simplified handwriting example
    # full_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=data_dir+'/test', transform=transform)
    test_dataset = datasets.ImageFolder(root=data_dir+'/test', transform=transform)

    total_size = len(train_dataset) + len(val_dataset)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_class_mapping():

    return {i: str(i) for i in range(2)}
