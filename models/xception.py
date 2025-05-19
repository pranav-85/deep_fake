import torch
import torch.nn as nn
import timm

class DeepFakeXception(nn.Module):
    """Xception-based model for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepFakeXception, self).__init__()
        self.model = timm.create_model("xception", pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


def train_xception(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """
    Train the Xception model for deepfake detection.
    
    Args:
        model: The Xception model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        lr: Learning rate
    
    Returns:
        model: Trained model
        history: Training history
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import numpy as np
    from tqdm import tqdm
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [], 
        'val_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=correct/total)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    return model, history


def evaluate_model(model, loader, device):
    """Evaluate model accuracy on the given data loader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total