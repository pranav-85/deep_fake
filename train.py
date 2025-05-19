from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all deepfake detection models.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass
    
    def train_epoch(self, train_loader, optimizer, device, scheduler=None):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for training
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += batch_size
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
            
        if scheduler:
            scheduler.step()
            
        return running_loss / total, correct / total
    
    def evaluate(self, val_loader, device):
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
            device: Device to evaluate on
            
        Returns:
            tuple: (validation_loss, accuracy)
        """
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                
                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                total += batch_size
                
        return running_loss / total, correct / total
    
    def predict(self, inputs, device):
        """
        Make predictions with the model.
        
        Args:
            inputs: Input tensor
            device: Device to predict on
            
        Returns:
            tuple: (predictions, probabilities)
        """
        self.eval()
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = self(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
        return preds, probs
    
    def fit(self, train_loader, val_loader, epochs, device, lr=1e-4, scheduler_t_max=10):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train for
            device: Device to train on
            lr: Learning rate
            scheduler_t_max: T_max for CosineAnnealingLR
            
        Returns:
            dict: Training history
        """
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_t_max)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, device, scheduler)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader, device)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
        return history