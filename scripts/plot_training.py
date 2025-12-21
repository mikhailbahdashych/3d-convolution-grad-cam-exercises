"""Plot training metrics from CSV files."""

import sys
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def plot_loss_curves(df, save_dir):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'loss_curves.png'}")
    plt.close()


def plot_accuracy_curves(df, save_dir):
    """Plot training and validation accuracy curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'accuracy_curves.png'}")
    plt.close()


def plot_learning_rate(df, save_dir):
    """Plot learning rate over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['learning_rate'], linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'learning_rate.png'}")
    plt.close()


def plot_combined_metrics(df, save_dir):
    """Plot all metrics in a single figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(df['epoch'], df['learning_rate'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Train/Val gap
    axes[1, 1].plot(df['epoch'], df['train_accuracy'] - df['val_accuracy'],
                    linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy Gap (Train - Val) (%)', fontsize=11)
    axes[1, 1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'combined_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'combined_metrics.png'}")
    plt.close()


def plot_per_class_accuracy(df, save_dir, top_n=None):
    """Plot per-class accuracy over epochs."""
    # Get class columns
    class_cols = [col for col in df.columns if col.startswith('class_')]

    if not class_cols:
        print("No per-class accuracy data found")
        return

    # If top_n is specified, plot only top N classes by final accuracy
    if top_n and top_n < len(class_cols):
        final_accs = df[class_cols].iloc[-1].dropna().sort_values(ascending=False)
        class_cols = final_accs.head(top_n).index.tolist()

    plt.figure(figsize=(14, 8))

    for col in class_cols:
        class_id = col.replace('class_', '')
        plt.plot(df['epoch'], df[col], label=f'Class {class_id}', linewidth=1.5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    title = 'Per-Class Accuracy Over Time'
    if top_n:
        title += f' (Top {top_n} Classes)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'per_class_accuracy{"_top" + str(top_n) if top_n else ""}.png'
    plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / filename}")
    plt.close()


def plot_per_class_heatmap(df, save_dir):
    """Plot per-class accuracy as a heatmap."""
    class_cols = [col for col in df.columns if col.startswith('class_')]

    if not class_cols:
        print("No per-class accuracy data found")
        return

    # Create heatmap data
    heatmap_data = df[class_cols].T
    heatmap_data.index = [col.replace('class_', '') for col in class_cols]

    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'}, annot=False)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Class ID', fontsize=12)
    plt.title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'per_class_heatmap.png'}")
    plt.close()


def plot_final_class_comparison(df, save_dir):
    """Plot final accuracy for each class as a bar chart."""
    class_cols = [col for col in df.columns if col.startswith('class_')]

    if not class_cols:
        print("No per-class accuracy data found")
        return

    # Get final accuracies
    final_accs = df[class_cols].iloc[-1].dropna()
    class_ids = [col.replace('class_', '') for col in final_accs.index]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_ids, final_accs.values, color='steelblue', alpha=0.8)

    # Color bars by performance
    for i, bar in enumerate(bars):
        if final_accs.values[i] >= 80:
            bar.set_color('green')
        elif final_accs.values[i] >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Final Per-Class Accuracy (Epoch {df["epoch"].iloc[-1]})',
              fontsize=14, fontweight='bold')
    plt.axhline(y=final_accs.mean(), color='black', linestyle='--',
                label=f'Mean: {final_accs.mean():.1f}%', linewidth=2)
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'final_class_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'final_class_comparison.png'}")
    plt.close()


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--metrics-dir', type=str, default='outputs/logs/metrics',
                        help='Directory containing metrics CSV files')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: same as metrics-dir)')
    parser.add_argument('--top-classes', type=int, default=None,
                        help='Plot only top N classes by accuracy')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    save_dir = Path(args.save_dir) if args.save_dir else metrics_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("TRAINING METRICS PLOTTING")
    print("="*60)
    print(f"Metrics directory: {metrics_dir}")
    print(f"Save directory: {save_dir}")

    # Load main metrics
    metrics_file = metrics_dir / 'training_metrics.csv'
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Please run training first to generate metrics.")
        return

    df = pd.read_csv(metrics_file)
    print(f"\nLoaded {len(df)} epochs of training data")

    # Generate plots
    print("\nGenerating plots...")

    plot_loss_curves(df, save_dir)
    plot_accuracy_curves(df, save_dir)
    plot_learning_rate(df, save_dir)
    plot_combined_metrics(df, save_dir)

    # Load and plot per-class accuracy
    per_class_file = metrics_dir / 'per_class_accuracy.csv'
    if per_class_file.exists():
        df_per_class = pd.read_csv(per_class_file)
        print(f"\nLoaded per-class accuracy data")

        plot_per_class_accuracy(df_per_class, save_dir, top_n=args.top_classes)
        if not args.top_classes:
            plot_per_class_accuracy(df_per_class, save_dir, top_n=10)
        plot_per_class_heatmap(df_per_class, save_dir)
        plot_final_class_comparison(df_per_class, save_dir)
    else:
        print(f"\nNo per-class accuracy file found: {per_class_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs: {len(df)}")
    print(f"Best training accuracy: {df['train_accuracy'].max():.2f}%")
    print(f"Best validation accuracy: {df['val_accuracy'].max():.2f}%")
    print(f"Final training loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Final learning rate: {df['learning_rate'].iloc[-1]:.2e}")

    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print(f"All plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
