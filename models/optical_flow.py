import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class OpticalFlowDataset(Dataset):
    """
    Dataset class for optical flow-based deepfake detection.
    Extracts and processes optical flow from videos.
    """
    def __init__(self, video_paths, labels=None, num_frames=10, transform=None):
        """
        Args:
            video_paths (list): List of paths to video files
            labels (list): List of labels (0=real, 1=fake)
            num_frames (int): Number of optical flow frames to extract
            transform (callable): Optional transform to apply to optical flow frames
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Extract optical flow frames
        cap = cv2.VideoCapture(video_path)
        ret, prev = cap.read()
        
        if not ret:
            raise ValueError(f"Could not read {video_path}")
        
        prev = cv2.resize(prev, (224, 224))
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        flow_frames = []
        
        frame_count = 0
        while len(flow_frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (224, 224))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            flow_frames.append(flow)
            prev_gray = gray
            frame_count += 1
            
        cap.release()
        
        # Handle videos with fewer frames than requested
        if len(flow_frames) < self.num_frames:
            # Pad with zeros if needed
            padding = self.num_frames - len(flow_frames)
            for _ in range(padding):
                flow_frames.append(np.zeros((224, 224, 2), dtype=np.float32))
                
        # Stack and convert to tensor
        flow_tensor = np.stack(flow_frames[:self.num_frames])
        flow_tensor = torch.tensor(flow_tensor).permute(0, 3, 1, 2).float()  # T, C, H, W
        
        if self.transform:
            flow_tensor = self.transform(flow_tensor)
            
        if self.labels is not None:
            label = self.labels[idx]
            return flow_tensor, label
        else:
            return flow_tensor


class OpticalFlowCNN(nn.Module):
    """
    CNN model for optical flow-based deepfake detection.
    Uses 3D convolutions to process temporal and spatial information.
    """
    def __init__(self, num_frames=10, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        
        # Calculate feature dimensions after conv layers
        feat_size = 64 * num_frames * (224 // 8) * (224 // 8)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) → reshape to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_optical_flow_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    """
    Train the optical flow model for deepfake detection.
    
    Args:
        model: The optical flow model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs to train
        lr: Learning rate
    
    Returns:
        model: Trained model
        history: Training history
    """
    from torch.optim import Adam
    from tqdm import tqdm
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
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
        
        for flows, labels in loop:
            flows, labels = flows.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(flows)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * flows.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=correct/total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        val_acc = evaluate_optical_flow_model(model, val_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    return model, history


def evaluate_optical_flow_model(model, loader, device):
    """Evaluate optical flow model accuracy on the given data loader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for flows, labels in loader:
            flows, labels = flows.to(device), labels.to(device)
            outputs = model(flows)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total