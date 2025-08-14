import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os


class TrainingResultsVisualizer:
    def __init__(self):
        """
        Class to visualize training results
        """
        self.results_dir = "training_visualizations"
        os.makedirs(self.results_dir, exist_ok=True)

        # Manually input your K-fold results
        self.kfold_results = {
            'Fold': [1, 2, 3, 4, 5],
            'Accuracy': [0.9190, 0.9213, 0.9187, 0.9358, 0.9353],
            'Precision': [0.9225, 0.9225, 0.9187, 0.9365, 0.9353],
            'Recall': [0.9190, 0.9213, 0.9187, 0.9358, 0.9353],
            'F1': [0.9187, 0.9187, 0.9187, 0.9353, 0.9353]
        }

        # Final results
        self.final_results = {
            'K-fold Average': {
                'Accuracy': 0.9190,
                'Precision': 0.9225,
                'Recall': 0.9190,
                'F1': 0.9187
            },
            'K-fold Std': {
                'Accuracy': 0.0218,
                'Precision': 0.0172,
                'Recall': 0.0218,
                'F1': 0.0220
            },
            'Best Model': {
                'Accuracy': 0.9358,
                'Precision': 0.9365,
                'Recall': 0.9358,
                'F1': 0.9353
            },
            'Final Test': {
                'Accuracy': 0.9213,
                'Precision': 0.9225,
                'Recall': 0.9213,
                'F1': 0.9187
            }
        }

        # Create DataFrame
        self.df = pd.DataFrame(self.kfold_results)
        print(f"Training visualizations will be saved to: {self.results_dir}")

    def plot_kfold_performance(self):
        """Visualize K-fold cross validation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Fold Cross Validation Results', fontsize=16, fontweight='bold')

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            row, col = i // 2, i % 2
            ax = axes[row, col]

            # Bar plot
            bars = ax.bar(self.df['Fold'], self.df[metric], color=color, alpha=0.7,
                          edgecolor='black', linewidth=1)

            # Display values on top of bars
            for bar, value in zip(bars, self.df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

            # Mean line
            mean_val = np.mean(self.df[metric])
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.8,
                       label=f'Mean: {mean_val:.4f}')

            ax.set_title(f'{metric} Across Folds', fontweight='bold')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric)
            ax.set_ylim(0.85, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Adjust X-axis
            ax.set_xticks(self.df['Fold'])

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.results_dir}/kfold_performance_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_metrics_comparison(self):
        """Compare all metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. All metrics across folds
        ax1 = axes[0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        x = np.arange(len(self.df['Fold']))
        width = 0.2

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax1.bar(x + i * width, self.df[metric], width, label=metric,
                    color=color, alpha=0.8)

        ax1.set_title('All Metrics Across Folds', fontweight='bold')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels([f'Fold {i}' for i in self.df['Fold']])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.85, 1.0)

        # 2. Final Results Comparison
        ax2 = axes[1]
        categories = ['K-fold Avg', 'Best Model', 'Final Test']

        accuracy_vals = [self.final_results['K-fold Average']['Accuracy'],
                         self.final_results['Best Model']['Accuracy'],
                         self.final_results['Final Test']['Accuracy']]

        f1_vals = [self.final_results['K-fold Average']['F1'],
                   self.final_results['Best Model']['F1'],
                   self.final_results['Final Test']['F1']]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, accuracy_vals, width, label='Accuracy',
                        color='#1f77b4', alpha=0.8)
        bars2 = ax2.bar(x + width / 2, f1_vals, width, label='F1 Score',
                        color='#d62728', alpha=0.8)

        # Display values on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_title('Final Results Comparison', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.85, 1.0)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.results_dir}/metrics_comparison_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_variance_analysis(self):
        """Variance and stability analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Box plot
        ax1 = axes[0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        data_for_box = [self.df[metric].values for metric in metrics]

        box_plot = ax1.boxplot(data_for_box, labels=metrics, patch_artist=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_title('Metrics Distribution Across Folds', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.85, 1.0)

        # 2. Standard deviation comparison
        ax2 = axes[1]
        std_values = []
        for metric in metrics:
            std_values.append(np.std(self.df[metric]))

        bars = ax2.bar(metrics, std_values, color=colors, alpha=0.8)

        # Display values on top of bars
        for bar, std_val in zip(bars, std_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                     f'{std_val:.4f}', ha='center', va='bottom', fontweight='bold')

        ax2.set_title('Standard Deviation Across Folds', fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.results_dir}/variance_analysis_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_table(self):
        """Create summary table"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        table_data = []

        # K-fold results
        for i, fold in enumerate(self.df['Fold']):
            table_data.append([
                f'Fold {fold}',
                f"{self.df.iloc[i]['Accuracy']:.4f}",
                f"{self.df.iloc[i]['Precision']:.4f}",
                f"{self.df.iloc[i]['Recall']:.4f}",
                f"{self.df.iloc[i]['F1']:.4f}"
            ])

        # Separator
        table_data.append(['---', '---', '---', '---', '---'])

        # Summary statistics
        table_data.extend([
            ['K-fold Mean',
             f"{self.final_results['K-fold Average']['Accuracy']:.4f}",
             f"{self.final_results['K-fold Average']['Precision']:.4f}",
             f"{self.final_results['K-fold Average']['Recall']:.4f}",
             f"{self.final_results['K-fold Average']['F1']:.4f}"],
            ['K-fold Std',
             f"{self.final_results['K-fold Std']['Accuracy']:.4f}",
             f"{self.final_results['K-fold Std']['Precision']:.4f}",
             f"{self.final_results['K-fold Std']['Recall']:.4f}",
             f"{self.final_results['K-fold Std']['F1']:.4f}"],
            ['Best Model',
             f"{self.final_results['Best Model']['Accuracy']:.4f}",
             f"{self.final_results['Best Model']['Precision']:.4f}",
             f"{self.final_results['Best Model']['Recall']:.4f}",
             f"{self.final_results['Best Model']['F1']:.4f}"],
            ['Final Test',
             f"{self.final_results['Final Test']['Accuracy']:.4f}",
             f"{self.final_results['Final Test']['Precision']:.4f}",
             f"{self.final_results['Final Test']['Recall']:.4f}",
             f"{self.final_results['Final Test']['F1']:.4f}"]
        ])

        # Create table
        table = ax.table(cellText=table_data,
                         colLabels=['Phase', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)

        # Header styling
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Best model row highlighting
        for i in range(5):
            table[(8, i)].set_facecolor('#90EE90')  # Light green for best model
            table[(9, i)].set_facecolor('#FFE4B5')  # Light orange for final test

        ax.set_title('Training Results Summary', fontweight='bold', pad=20, fontsize=14)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'{self.results_dir}/summary_table_{timestamp}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

    def run_all_visualizations(self):
        """Run all visualizations"""
        print("=" * 60)
        print("TRAINING RESULTS VISUALIZATION")
        print("=" * 60)

        print("Using manual data for visualization...")
        self.plot_kfold_performance()
        self.plot_metrics_comparison()
        self.plot_variance_analysis()
        self.create_summary_table()

        print(f"\nAll training visualizations saved to: {self.results_dir}")
        print("=" * 60)


# Usage
if __name__ == "__main__":
    visualizer = TrainingResultsVisualizer()
    visualizer.run_all_visualizations()
