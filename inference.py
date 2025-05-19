import torch
import torch.nn as nn
import torchvision.models as models

def get_vit_model(num_classes=2, pretrained=True):
    """
    Creates a Vision Transformer model for deepfake detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use pretrained weights (default: True)
        
    Returns:
        torch.nn.Module: The ViT model
    """
    vit = models.vision_transformer.vit_b_16(pretrained=pretrained)
    
    # Modify the final classification head for binary classification
    in_features = vit.heads[0].in_features
    vit.heads = nn.Linear(in_features, num_classes)
    
    return vit


class DeepFakeViT(nn.Module):
    """Vision Transformer model for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(DeepFakeViT, self).__init__()
        self.model = get_vit_model(num_classes, pretrained)
    
    def forward(self, x):
        return self.model(x)


def train_vit(model, train_loader, val_loader, device, epochs=10, lr=1e-5):
    """
    Train the Vision Transformer model for deepfake detection.
    
    Args:
        model: The ViT model
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
        
        for images, labels in train_loader:
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