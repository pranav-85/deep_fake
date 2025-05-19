import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

def compute_eer(fpr, tpr):
    """Compute Equal Error Rate (EER) from FPR and TPR.
    
    Args:
        fpr (array): False positive rates.
        tpr (array): True positive rates.
        
    Returns:
        eer (float): Equal Error Rate.
    """
    # Find the point where FPR is closest to 1 - TPR
    eer_threshold = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_threshold]
    return eer

def plot_metrics(all_labels, all_probs, all_preds):
    """Plot evaluation metrics including ROC curve, PR curve, and confusion matrix.
    
    Args:
        all_labels (array): Ground truth labels.
        all_probs (array): Predicted probabilities.
        all_preds (array): Predicted labels.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(all_labels, all_probs):.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set.
    
    Args:
        model: Model to evaluate.
        test_loader: DataLoader for test dataset.
        device: Device to run evaluation on.
        
    Returns:
        acc (float): Accuracy.
        prec (float): Precision.
        auc (float): Area under ROC curve.
        eer (float): Equal Error Rate.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Prob of class "Fake"
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    eer = compute_eer(fpr, tpr)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"EER: {eer:.4f}")

    plot_metrics(all_labels, all_probs, all_preds)
    
    return acc, prec, auc, eer

def compare_models(model_results, metric_names=['Accuracy', 'Precision', 'AUC', 'EER']):
    """Compare performance of multiple models.
    
    Args:
        model_results (dict): Dictionary mapping model names to metric values.
        metric_names (list): List of metric names.
    """
    models = list(model_results.keys())
    
    fig, axs = plt.subplots(len(metric_names), 1, figsize=(10, 3*len(metric_names)))
    
    for i, metric in enumerate(metric_names):
        values = [model_results[model][i] for model in models]
        
        # For EER, lower is better
        if metric == 'EER':
            values = [1 - val for val in values]  # Invert for visualization
            metric = 'EER (lower is better)'
            
        axs[i].bar(models, values, color=['skyblue', 'lightgreen', 'salmon', 'lightpurple'][:len(models)])
        axs[i].set_title(f'{metric} Comparison')
        axs[i].set_ylim(0, 1)
        
        # Add value labels
        for j, v in enumerate(values):
            if metric == 'EER (lower is better)':
                v = 1 - v  # Revert for display
            axs[i].text(j, v + 0.02, f'{v:.4f}', ha='center')
            
    plt.tight_layout()
    plt.show()