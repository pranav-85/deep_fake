import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

def compute_eer(fpr, tpr):
    """
    Compute the Equal Error Rate (EER) from ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        
    Returns:
        eer: Equal Error Rate value
    """
    # Find the point where FPR is closest to 1 - TPR
    fnr = 1 - tpr
    eer_threshold = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_threshold]
    return eer


def evaluate_model_metrics(model, test_loader, device):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to use (cuda/cpu)
        
    Returns:
        metrics: Dictionary of metrics
        all_labels: Ground truth labels
        all_preds: Model predictions
        all_probs: Model prediction probabilities
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of "Fake" class
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    # Compute EER
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    metrics['eer'] = compute_eer(fpr, tpr)
    
    return metrics, all_labels, all_preds, all_probs


def plot_metrics(all_labels, all_probs, all_preds, model_name=None):
    """
    Plot evaluation metrics: ROC curve, PR curve, and confusion matrix.
    
    Args:
        all_labels: Ground truth labels
        all_probs: Model prediction probabilities
        all_preds: Model predictions
        model_name: Name of the model for plot titles
    """
    model_prefix = f"{model_name} " if model_name else ""
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    eer = compute_eer(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(all_labels, all_probs):.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.plot([eer], [1-eer], 'ro', label=f"EER = {eer:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_prefix}ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_prefix}Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_prefix}Confusion Matrix")
    plt.show()


def print_metrics_report(metrics, model_name=None):
    """
    Print a formatted report of model metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model for the report title
    """
    title = f"{model_name} " if model_name else ""
    
    print(f"\n{title}Performance Metrics")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print(f"EER:       {metrics['eer']:.4f}")
    print("=" * 50)


def compare_models(model_results, names):
    """
    Compare the performance of multiple models.
    
    Args:
        model_results: List of model metrics dictionaries
        names: List of model names
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'eer']
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.8 / len(model_results)
    
    for i, (result, name) in enumerate(zip(model_results, names)):
        values = [result[metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=name)
    
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * (len(model_results) - 1) / 2, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()