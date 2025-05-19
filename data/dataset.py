import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class DatasetLoader(Dataset):
    def __init__(self, root_dir, transform=None, balance_data=False):
        """
        Dataset loader for deepfake detection.
        
        Args:
            root_dir (str): Root directory of the dataset (containing 'real' and 'fake' folders).
            transform (callable, optional): Transform to be applied on the images.
            balance_data (bool, optional): Whether to balance real and fake samples.
        """
        self.real_samples = []
        self.fake_samples = []
        self.labels = []
        self.transform = transform

        fake_by_identity = defaultdict(list)

        for label in ['real', 'fake']:
            class_dir = os.path.join(root_dir, label)
            if not os.path.exists(class_dir):
                continue
                
            for video_folder in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_folder)

                if not os.path.isdir(video_path):
                    continue

                for frame_name in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame_name)
                    if frame_path.endswith(('.jpg', '.png')):
                        if label == 'real':
                            self.real_samples.append((frame_path, 0))
                        else:
                            target_id = video_folder.split('_')[0]
                            fake_by_identity[target_id].append((frame_path, 1))

        if balance_data and self.real_samples and fake_by_identity:
            num_real = len(self.real_samples)
            num_identities = len(fake_by_identity)
            samples_per_identity = max(1, num_real // num_identities)

            balanced_fake_samples = []
            for identity, samples in fake_by_identity.items():
                if len(samples) >= samples_per_identity:
                    balanced_fake_samples.extend(random.sample(samples, samples_per_identity))
                else:
                    balanced_fake_samples.extend(samples)  

            self.fake_samples = balanced_fake_samples[:num_real]  
        else:
            for samples in fake_by_identity.values():
                self.fake_samples.extend(samples)

        self.samples = self.real_samples + self.fake_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)
    
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        image = image.float()

        return image, label


class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None):
        """Dataset for loading frames from a single video directory."""
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


class OpticalFlowDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_flow_frames=10):
        """
        Dataset for optical flow-based deepfake detection.
        
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Transform to be applied on optical flow.
            num_flow_frames (int): Number of optical flow frames to use.
        """
        self.samples = []
        self.transform = transform
        self.num_flow_frames = num_flow_frames

        for label, cls in enumerate(['real', 'fake']):
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                if fname.endswith('.mp4'):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(video_path)
        ret, prev = cap.read()

        if not ret:
            raise ValueError(f"Could not read {video_path}")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        flow_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.resize(flow, (224, 224))  # resize for model input
            flow_frames.append(flow)
            prev_gray = gray

            if len(flow_frames) >= self.num_flow_frames:
                break

        cap.release()

        # Handle case when we don't have enough frames
        if not flow_frames:
            # Return zeros if no frames are available
            flow_tensor = torch.zeros((self.num_flow_frames, 2, 224, 224))
            return flow_tensor, label
        
        # Pad if we don't have enough frames
        while len(flow_frames) < self.num_flow_frames:
            flow_frames.append(np.zeros_like(flow_frames[0]))

        # Stack frames
        flow_tensor = np.stack(flow_frames[:self.num_flow_frames])  
        flow_tensor = torch.tensor(flow_tensor).permute(0, 3, 1, 2).float()  # T, C, H, W

        if self.transform:
            flow_tensor = self.transform(flow_tensor)

        return flow_tensor, label