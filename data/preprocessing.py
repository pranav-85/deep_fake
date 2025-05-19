import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
from collections import defaultdict


def extract_frames(video_path, num_frames=30):
    '''Function to extract frames from the video file.
        Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
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


def save_frames(dataset_path):
    '''Function to save the extracted frames to a directory.
        Args:
        dataset_path (str): Path to the dataset directory.
    '''
    
    videos = os.listdir(dataset_path)

    for video in videos:
        if not video.endswith(('.mp4', '.avi', '.mov')):
            continue
            
        os.makedirs(f"{dataset_path}/{video.split('.')[0]}", exist_ok=True)
        video_path = os.path.join(dataset_path, video)
        frames = extract_frames(video_path, num_frames=30)

        for i, frame in enumerate(frames):
            plt.imsave(f"{dataset_path}/{video.split('.')[0]}/frame-{i}.jpg", frame)

        print(f"Frames extracted and saved for {video}")


def merge_directories(dir1, dir2, merged_dir):
    '''Merge two directories into one.
        Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        merged_dir (str): Path to the merged directory.
    '''
    os.makedirs(merged_dir, exist_ok=True)

    for src_dir in [dir1, dir2]:
        for root, _, files in os.walk(src_dir):
            for file in files:
                src_path = os.path.join(root, file)
                
                # Optional: keep subfolders or flatten
                relative_path = os.path.relpath(root, src_dir)
                dest_folder = os.path.join(merged_dir, relative_path)
                os.makedirs(dest_folder, exist_ok=True)

                dest_path = os.path.join(dest_folder, file)
                
                if not os.path.exists(dest_path):  # avoid overwriting
                    shutil.copy2(src_path, dest_path)
                else:
                    # Rename if duplicate
                    base, ext = os.path.splitext(file)
                    new_filename = f"{base}_copy{ext}"
                    shutil.copy2(src_path, os.path.join(dest_folder, new_filename))

    print(f"[✔] Merged {dir1} and {dir2} into {merged_dir}")


def split_dataset(base_dir, output_dir=None, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    Splits the dataset in base_dir into train/val/test and copies to output_dir.
    
    Args:
        base_dir (str): Path to the base directory containing 'real' and 'fake' folders.
        output_dir (str): Path to output the split dataset.
        split_ratio (tuple): Ratio for train/val/test split.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    output_dir = output_dir or (base_dir + '_split')
    categories = ['real', 'fake']

    for category in categories:
        category_dir = os.path.join(base_dir, category)
        videos = [d for d in os.listdir(category_dir) if os.path.isdir(os.path.join(category_dir, d))]
        random.shuffle(videos)

        n_total = len(videos)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]

        for split_name, split_videos in zip(['train', 'val', 'test'], [train_videos, val_videos, test_videos]):
            split_dir = os.path.join(output_dir, split_name, category)
            os.makedirs(split_dir, exist_ok=True)
            for vid in split_videos:
                src = os.path.join(category_dir, vid)
                dst = os.path.join(split_dir, vid)
                if os.path.exists(dst):
                    continue
                shutil.copytree(src, dst)
    
    print(f"[✔] Dataset split into train/val/test at: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess deepfake detection dataset')
    parser.add_argument('--real_dir', type=str, default='real', help='Directory with real videos')
    parser.add_argument('--fake_dir', type=str, default='fake', help='Directory with fake videos')
    parser.add_argument('--output_dir', type=str, default='split_dataset', help='Output directory for split dataset')
    parser.add_argument('--extract_frames', action='store_true', help='Extract frames from videos')
    
    args = parser.parse_args()
    
    if args.extract_frames:
        print("Extracting frames from real videos...")
        save_frames(args.real_dir)
        print("Extracting frames from fake videos...")
        save_frames(args.fake_dir)
    
    print("Splitting dataset...")
    split_dataset(os.getcwd(), args.output_dir)
    print("Done!")
