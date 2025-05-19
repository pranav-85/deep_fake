import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def extract_frames(video_path, num_frames=30):
    '''Function to extract frames from the video file.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
    Returns:
        frames (list): List of extracted frames.
    '''
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = list(np.linspace(0, frame_count-1, num_frames, dtype=int))

    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i in frame_idxs:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to 224x224
            frame = cv2.resize(frame, (224, 224))
            # Normalize frame
            frame = frame / 255.0
            frames.append(frame)
        
    cap.release()
    return frames

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None):
        """Dataset for loading frames from a directory.
        
        Args:
            frame_dir (str): Directory containing frames.
            transform: Optional transform to apply to the frames.
        """
        self.frame_paths = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith((".jpg", ".png"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.frame_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).float() / 255.0  # Convert to float tensor
            image = image.permute(2, 0, 1)  # HWC -> CHW
        return image

def predict_from_frames(frame_dir, model, device, transform=None):
    """Predict deepfake probability from frames in a directory.
    
    Args:
        frame_dir (str): Directory containing frames.
        model: The model to use for prediction.
        device: Device to run the model on.
        transform: Optional transform to apply to the frames.
        
    Returns:
        prediction (int): 0 for real, 1 for fake.
        confidence (float): Probability of being fake.
    """
    dataset = FrameDataset(frame_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of being fake
            all_probs.extend(probs.cpu().numpy())

    avg_prob = np.mean(all_probs)
    prediction = 1 if avg_prob >= 0.5 else 0
    return prediction, avg_prob

def detect_deepfake(video_path, model, device, temp_dir='temp_frames'):
    """Detect if a video is deepfake.
    
    Args:
        video_path (str): Path to the video file.
        model: The model to use for prediction.
        device: Device to run the model on.
        temp_dir (str): Directory to store temporary frames.
        
    Returns:
        label (str): "Real" or "Fake".
        confidence (float): Probability of being fake.
    """
    frames = extract_frames(video_path, num_frames=30)

    # Save the frames in a directory
    os.makedirs(temp_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        plt.imsave(f"{temp_dir}/frame-{i}.jpg", frame)

    prediction, confidence = predict_from_frames(temp_dir, model, device)

    label = "Fake" if prediction else "Real"
    return label, confidence

def visualize_results(frames, prediction, confidence, num_frames_to_show=6):
    """Visualize prediction results with frames from the video.
    
    Args:
        frames (list): List of frames from the video.
        prediction (str): "Real" or "Fake" prediction.
        confidence (float): Prediction confidence.
        num_frames_to_show (int): Number of frames to display.
    """
    fig, axes = plt.subplots(1, num_frames_to_show, figsize=(15, 3))
    
    frame_indices = np.linspace(0, len(frames) - 1, num_frames_to_show, dtype=int)
    
    for i, idx in enumerate(frame_indices):
        axes[i].imshow(frames[idx])
        axes[i].axis('off')
    
    fig.suptitle(f"Prediction: {prediction} (Confidence: {confidence:.4f})", fontsize=16)
    plt.tight_layout()
    plt.show()