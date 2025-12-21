"""Evaluation metrics for model performance."""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class indices (B,)
        targets: Ground truth class indices (B,)

    Returns:
        Accuracy as percentage
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model outputs (B, C)
        targets: Ground truth labels (B,)
        k: Top-k classes to consider

    Returns:
        Top-k accuracy as percentage
    """
    _, top_k_preds = logits.topk(k, dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 17,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(targets, predictions, labels=labels)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        save_path: Path to save figure
        normalize: If True, normalize by row (true labels)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        ax=ax,
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 17,
) -> Dict[int, Dict[str, float]]:
    """
    Compute precision, recall, F1-score for each class.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to metrics dict
    """
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )

    per_class_metrics = {}
    for class_id in range(num_classes):
        per_class_metrics[class_id] = {
            'precision': precision[class_id] * 100,
            'recall': recall[class_id] * 100,
            'f1_score': f1[class_id] * 100,
            'support': int(support[class_id]),
        }

    return per_class_metrics


def compute_mean_class_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 17,
) -> float:
    """
    Compute mean per-class accuracy (accounts for class imbalance).

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes

    Returns:
        Mean class accuracy as percentage
    """
    cm = compute_confusion_matrix(predictions, targets, num_classes)

    per_class_acc = []
    for i in range(num_classes):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            per_class_acc.append(acc)

    return 100.0 * np.mean(per_class_acc)


def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 17,
):
    """
    Print detailed classification report.

    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        num_classes: Number of classes
    """
    labels = list(range(num_classes))
    target_names = [f"Class {i}" for i in range(num_classes)]

    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)


class MetricsTracker:
    """Track metrics during evaluation."""

    def __init__(self, num_classes: int = 17):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_logits = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch results.

        Args:
            logits: Model outputs (B, C)
            targets: Ground truth labels (B,)
        """
        predictions = logits.argmax(dim=1)

        self.all_predictions.append(predictions.cpu().numpy())
        self.all_targets.append(targets.cpu().numpy())
        self.all_logits.append(logits.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metrics
        """
        # Concatenate all batches
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        logits = np.concatenate(self.all_logits)

        # Convert to tensors for top-k accuracy
        logits_tensor = torch.from_numpy(logits)
        targets_tensor = torch.from_numpy(targets)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions) * 100,
            'mean_class_accuracy': compute_mean_class_accuracy(predictions, targets, self.num_classes),
            'top_3_accuracy': compute_top_k_accuracy(logits_tensor, targets_tensor, k=3),
        }

        # Per-class metrics
        per_class = compute_per_class_metrics(predictions, targets, self.num_classes)

        # Add mean metrics
        metrics['mean_precision'] = np.mean([m['precision'] for m in per_class.values()])
        metrics['mean_recall'] = np.mean([m['recall'] for m in per_class.values()])
        metrics['mean_f1'] = np.mean([m['f1_score'] for m in per_class.values()])

        return metrics, per_class

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        return compute_confusion_matrix(predictions, targets, self.num_classes)

    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all predictions and targets."""
        predictions = np.concatenate(self.all_predictions)
        targets = np.concatenate(self.all_targets)
        return predictions, targets
