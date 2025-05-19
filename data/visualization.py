import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import seaborn as sns


def plot_class_distribution(dataset_path):
    """
    Plot the distribution of real and fake samples in the dataset.
    
    Args:
        dataset_path (str): Path to the dataset directory.
    """
    num_real = len(os.listdir(os.path.join(dataset_path, 'real')))
    num_fake = len(os.listdir(os.path.join(dataset_path, 'fake')))

    plt.figure(figsize=(10, 5))
    plt.bar(['Real', 'Fake'], [num_real, num_fake], color=['blue', 'red'])
    plt.title('Number of Samples')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')

    print(f"Number of samples: {num_real + num_fake}")
    print(f"Number of real samples: {num_real}")
    print(f"Number of fake samples: {num_fake}")
    
    return plt


def plot_training_distribution(train_dataset):
    """
    Plot the distribution of classes in the training dataset.
    
    Args:
        train_dataset: The training dataset.
    """
    # Count labels from the train dataset
    labels = [label for _, label in train_dataset]
    counts = Counter(labels)

    # Map labels to class names
    class_names = {0: 'Real', 1: 'Fake'}
    labels_text = [class_names[int(label)] for label in counts.keys()]
    values = list(counts.values())

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(labels_text, values, color=['skyblue', 'salmon'])
    plt.title('Number of Real vs Fake Samples in Training Data')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt


def display_sample_frames(video_path, num_frames=30):
    """
    Display sample frames from a video.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to display.
    """
    from data.preprocessing import extract_frames
    
    frames = extract_frames(video_path, num_frames)
    
    plt.figure(figsize=(15, 10))
    for i in range(len(frames)):
        plt.subplot(5, 6, i+1)
        plt.imshow(frames[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return plt


def plot_confusion_matrix(cm, class_names=['Real', 'Fake']):
    """
    Plot a confusion matrix.
    
    Args:
        cm (array): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    return plt


def plot_roc_curve(fpr, tpr, auc_score):
    """
    Plot an ROC curve.
    
    Args:
        fpr (array): False positive rate.
        tpr (array): True positive rate.
        auc_score (float): Area under the curve.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    
    return plt


def plot_precision_recall_curve(precision, recall):
    """
    Plot a precision-recall curve.
    
    Args:
        precision (array): Precision values.
        recall (array): Recall values.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    
    return plt