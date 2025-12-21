"""Trainer for exercise recognition model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time
import csv
from typing import Dict, Optional

from ..utils.checkpointing import save_checkpoint
from ..utils.logging import setup_logger


class Trainer:
    """Trainer class for model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger=None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            config: Configuration dict
            scheduler: Optional learning rate scheduler
            logger: Optional logger
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.logger = logger or setup_logger("trainer")

        # Training settings
        self.num_epochs = config.get("training.num_epochs", 100)
        self.grad_clip = config.get("training.gradient_clipping.max_norm", 1.0)
        self.log_freq = config.get("logging.log_freq", 10)

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint.save_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = config.get("checkpoint.save_freq", 10)

        # Metrics directory for CSV files
        self.metrics_dir = Path(config.get("logging.log_dir", "outputs/logs")) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = config.get("training.early_stopping.patience", 10)
        self.early_stopping_min_delta = config.get("training.early_stopping.min_delta", 0.001)

        # TensorBoard
        if config.get("logging.tensorboard", True):
            log_dir = Path(config.get("logging.log_dir", "outputs/logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            self.writer = None

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        self.learning_rates = []
        self.per_class_accuracies = []  # List of dicts, one per epoch

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch_idx, (clips, labels, metadata) in enumerate(pbar):
            # Move to device
            clips = clips.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(clips)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            if batch_idx % self.log_freq == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%"
                })

        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate model.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Per-class accuracy
        num_classes = self.config.get("model.num_classes", 17)
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for clips, labels, metadata in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                clips = clips.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(clips)
                loss = self.criterion(outputs, labels)

                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        # Compute metrics
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total

        # Per-class accuracy
        per_class_acc = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                per_class_acc[i] = 100.0 * class_correct[i] / class_total[i]

        return {
            "loss": val_loss,
            "accuracy": val_acc,
            "per_class_accuracy": per_class_acc,
        }

    def save_metrics(self):
        """Save training metrics to CSV files."""
        # Save main metrics (loss, accuracy, learning rate)
        metrics_file = self.metrics_dir / "training_metrics.csv"
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
            ])
            for i in range(len(self.train_losses)):
                writer.writerow([
                    i + 1,
                    self.train_losses[i],
                    self.train_accuracies[i] if i < len(self.train_accuracies) else "",
                    self.val_losses[i] if i < len(self.val_losses) else "",
                    self.val_accuracies[i] if i < len(self.val_accuracies) else "",
                    self.learning_rates[i] if i < len(self.learning_rates) else "",
                ])

        # Save per-class accuracy
        if self.per_class_accuracies:
            per_class_file = self.metrics_dir / "per_class_accuracy.csv"
            num_classes = self.config.get("model.num_classes", 17)
            with open(per_class_file, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["epoch"] + [f"class_{i}" for i in range(num_classes)]
                writer.writerow(header)
                for epoch, class_accs in enumerate(self.per_class_accuracies):
                    row = [epoch + 1]
                    for i in range(num_classes):
                        row.append(class_accs.get(i, ""))
                    writer.writerow(row)

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            self.train_accuracies.append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics["loss"])
            self.val_accuracies.append(val_metrics["accuracy"])
            self.per_class_accuracies.append(val_metrics["per_class_accuracy"])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            # Save metrics to CSV after each epoch
            self.save_metrics()

            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            # Log per-class accuracy
            if epoch % 5 == 0:
                self.logger.info("Per-class validation accuracy:")
                for class_id, acc in val_metrics["per_class_accuracy"].items():
                    self.logger.info(f"  Class {class_id}: {acc:.2f}%")

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)

                # Log per-class accuracy
                for class_id, acc in val_metrics["per_class_accuracy"].items():
                    self.writer.add_scalar(f"Accuracy/class_{class_id}", acc, epoch)

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Learning_rate", current_lr, epoch)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Save checkpoints
            is_best_loss = val_metrics["loss"] < self.best_val_loss
            is_best_acc = val_metrics["accuracy"] > self.best_val_acc

            if is_best_loss:
                self.best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
                    str(self.checkpoint_dir),
                    "best_model_loss.pth",
                    config=self.config,
                )
                self.logger.info(f"+ Saved best model (loss: {val_metrics['loss']:.4f})")

            if is_best_acc:
                self.best_val_acc = val_metrics["accuracy"]
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
                    str(self.checkpoint_dir),
                    "best_model_acc.pth",
                    config=self.config,
                )
                self.logger.info(f"+ Saved best model (acc: {val_metrics['accuracy']:.2f}%)")

            # Periodic checkpoint
            if (epoch + 1) % self.save_freq == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    {"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]},
                    str(self.checkpoint_dir),
                    f"checkpoint_epoch_{epoch + 1}.pth",
                    config=self.config,
                )

            # Early stopping
            if is_best_acc:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        # Training complete
        elapsed_time = time.time() - start_time
        self.logger.info(f"\nTraining complete in {elapsed_time / 3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        if self.writer:
            self.writer.close()

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
        }
