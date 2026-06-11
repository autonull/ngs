import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_dataloaders(batch_size=64, root='./lean_ngs/data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_split_mnist_dataloaders(batch_size=64, root='./lean_ngs/data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
    
    tasks_train = []
    tasks_test = []
    
    # Create 5 tasks, each with 2 digits: (0,1), (2,3), (4,5), (6,7), (8,9)
    for t in range(5):
        digits = [2*t, 2*t + 1]
        
        train_idx = [i for i, (img, target) in enumerate(train_dataset) if target in digits]
        test_idx = [i for i, (img, target) in enumerate(test_dataset) if target in digits]
        
        task_train_ds = Subset(train_dataset, train_idx)
        task_test_ds = Subset(test_dataset, test_idx)
        
        tasks_train.append(DataLoader(task_train_ds, batch_size=batch_size, shuffle=True))
        tasks_test.append(DataLoader(task_test_ds, batch_size=batch_size, shuffle=False))
        
    return tasks_train, tasks_test

if __name__ == '__main__':
    print("Testing MNIST dataloaders...")
    train, test = get_mnist_dataloaders()
    print(f"MNIST train batches: {len(train)}")
    
    print("Testing Split-MNIST dataloaders...")
    tasks_train, tasks_test = get_split_mnist_dataloaders()
    for i, loader in enumerate(tasks_train):
        print(f"Split-MNIST Task {i} train batches: {len(loader)}")
