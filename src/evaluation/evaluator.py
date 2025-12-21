"""Model evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .metrics import MetricsTracker, plot_confusion_matrix, print_classification_report


class Evaluator:
    """Evaluator for model performance."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        num_classes: int = 17,
        save_dir: str = "outputs/results",
    ):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model
            data_loader: Data loader for evaluation
            device: Device to run on
            num_classes: Number of classes
            save_dir: Directory to save results
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_tracker = MetricsTracker(num_classes)

    def evaluate(self, save_predictions: bool = True) -> Dict:
        """
        Evaluate model on dataset.

        Args:
            save_predictions: If True, save predictions to CSV

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        self.metrics_tracker.reset()

        print(f"Evaluating on {len(self.data_loader.dataset)} samples...")

        with torch.no_grad():
            for clips, labels, metadata in tqdm(self.data_loader, desc="Evaluation"):
                clips = clips.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(clips)

                # Update metrics
                self.metrics_tracker.update(logits, labels)

        # Compute metrics
        metrics, per_class_metrics = self.metrics_tracker.compute()

        # Print results
        self._print_results(metrics, per_class_metrics)

        # Save results
        self._save_results(metrics, per_class_metrics)

        # Save predictions
        if save_predictions:
            self._save_predictions()

        # Save confusion matrix
        self._save_confusion_matrix()

        return metrics

    def _print_results(self, metrics: Dict, per_class_metrics: Dict):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy:      {metrics['accuracy']:.2f}%")
        print(f"Mean Class Accuracy:   {metrics['mean_class_accuracy']:.2f}%")
        print(f"Top-3 Accuracy:        {metrics['top_3_accuracy']:.2f}%")
        print(f"Mean Precision:        {metrics['mean_precision']:.2f}%")
        print(f"Mean Recall:           {metrics['mean_recall']:.2f}%")
        print(f"Mean F1-Score:         {metrics['mean_f1']:.2f}%")

        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-"*60)

        for class_id, class_metrics in per_class_metrics.items():
            print(
                f"{class_id:<8} "
                f"{class_metrics['precision']:>10.2f}%  "
                f"{class_metrics['recall']:>10.2f}%  "
                f"{class_metrics['f1_score']:>10.2f}%  "
                f"{class_metrics['support']:>8}"
            )

        # Print classification report
        predictions, targets = self.metrics_tracker.get_predictions()
        print_classification_report(predictions, targets, self.num_classes)

    def _save_results(self, metrics: Dict, per_class_metrics: Dict):
        """Save metrics to files."""
        # Save overall metrics
        metrics_file = self.save_dir / "metrics.txt"
        with open(metrics_file, "w") as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy:      {metrics['accuracy']:.2f}%\n")
            f.write(f"Mean Class Accuracy:   {metrics['mean_class_accuracy']:.2f}%\n")
            f.write(f"Top-3 Accuracy:        {metrics['top_3_accuracy']:.2f}%\n")
            f.write(f"Mean Precision:        {metrics['mean_precision']:.2f}%\n")
            f.write(f"Mean Recall:           {metrics['mean_recall']:.2f}%\n")
            f.write(f"Mean F1-Score:         {metrics['mean_f1']:.2f}%\n")

        # Save per-class metrics to CSV
        per_class_df = pd.DataFrame(per_class_metrics).T
        per_class_csv = self.save_dir / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_csv, index_label="class_id")

        print(f"\n✓ Metrics saved to {self.save_dir}")

    def _save_predictions(self):
        """Save predictions to CSV."""
        predictions, targets = self.metrics_tracker.get_predictions()

        df = pd.DataFrame({
            'true_label': targets,
            'predicted_label': predictions,
        })

        predictions_file = self.save_dir / "predictions.csv"
        df.to_csv(predictions_file, index=False)

        print(f"✓ Predictions saved to {predictions_file}")

    def _save_confusion_matrix(self):
        """Save confusion matrix plots."""
        cm = self.metrics_tracker.get_confusion_matrix()

        # Save counts
        cm_counts_file = self.save_dir / "confusion_matrix_counts.png"
        plot_confusion_matrix(cm, str(cm_counts_file), normalize=False, title="Confusion Matrix (Counts)")

        # Save normalized
        cm_normalized_file = self.save_dir / "confusion_matrix_normalized.png"
        plot_confusion_matrix(cm, str(cm_normalized_file), normalize=True, title="Confusion Matrix (Normalized)")

        # Save as CSV
        cm_csv = self.save_dir / "confusion_matrix.csv"
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(cm_csv, index=True, header=True)

        print(f"✓ Confusion matrices saved to {self.save_dir}")

    def analyze_errors(self, top_k: int = 10):
        """
        Analyze top-k misclassified examples.

        Args:
            top_k: Number of worst errors to analyze
        """
        predictions, targets = self.metrics_tracker.get_predictions()

        # Find misclassified samples
        misclassified_idx = np.where(predictions != targets)[0]

        if len(misclassified_idx) == 0:
            print("No misclassified samples!")
            return

        print(f"\nFound {len(misclassified_idx)} misclassified samples")
        print(f"Misclassification rate: {100 * len(misclassified_idx) / len(targets):.2f}%")

        # Analyze by class
        print("\nMisclassifications by true class:")
        for class_id in range(self.num_classes):
            class_mask = targets == class_id
            class_misclassified = np.sum((predictions != targets) & class_mask)
            class_total = np.sum(class_mask)

            if class_total > 0:
                error_rate = 100 * class_misclassified / class_total
                print(f"  Class {class_id}: {class_misclassified}/{class_total} ({error_rate:.2f}%)")
