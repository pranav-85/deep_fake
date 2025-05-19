import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import timm
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import shutil

# Import custom modules
from data.dataset import DatasetLoader
from evaluate import evaluate_model

def set_seed(seed=42):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(model_type="resnet"):
    """Get model based on model type.
    
    Args:
        model_type (str): Type of model to use.
        
    Returns:
        model: The model.
    """
    if model_type == "resnet":
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_type == "vit":
        model = models.vision_transformer.vit_b_16(pretrained=True)
        in_features = model.heads[0].in_features
        model.heads = nn.Linear(in_features, 2)
    elif model_type == "xception":
        model = timm.create_model("xception", pretrained=True, num_classes=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """Train model.
    
    Args:
        model: Model to train.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs (int): Number of epochs to train for.
        device: Device to train on.
        
    Returns:
        model: The trained model.
        history (dict): Training history.
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })

        train_loss = running_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += batch_size
        
        val_loss = running_loss / total
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    return model, history

def plot_training_history(history):
    """Plot training history.
    
    Args:
        history (dict): Training history.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="DeepFake Detection Training")
    parser.add_argument("--data_dir", type=str, default="split_dataset", help="Path to dataset directory")
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "vit", "xception"], help="Type of model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for saving models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    print("Loading datasets...")
    train_dataset = DatasetLoader(train_dir, common_transform=None, balance_data=True)
    val_dataset = DatasetLoader(val_dir, common_transform=None, balance_data=True)
    test_dataset = DatasetLoader(test_dir, common_transform=None, balance_data=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Count class distribution
    train_labels = [label for _, label in train_dataset]
    counts = Counter(train_labels)
    print(f"Class distribution in training set: {counts}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = get_model(args.model_type)
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Train model
    print("Beginning training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    evaluate_model(model, test_loader, device)
    
    # Save model
    model_filename = f"{args.model_type}_deepfake_model.pth"
    model_path = os.path.join(args.output_dir, model_filename)
    if args.model_type == "resnet":
        torch.save(model, model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()