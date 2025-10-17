"""
Visualization Module for Gait-Based Biometric Recognition
Provides comprehensive visualization tools for gait analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GaitVisualizer:
    """
    Comprehensive visualization tools for gait biometric analysis
    """
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
        
    def plot_gait_cycle(self, cycle_data, subject_id=None, save_path=None):
        """
        Plot a single gait cycle showing all three acceleration components
        
        Args:
            cycle_data (dict): Dictionary containing gait cycle data
            subject_id (str): Subject identifier
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        time = cycle_data['time']
        signals = ['vertical_acc', 'ap_acc', 'ml_acc']
        titles = ['Vertical Acceleration', 'Anterior-Posterior Acceleration', 'Medial-Lateral Acceleration']
        colors = ['red', 'blue', 'green']
        
        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            axes[i].plot(time, cycle_data[signal], color=color, linewidth=2)
            axes[i].set_title(f'{title}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Time (normalized)')
            axes[i].set_ylabel('Acceleration (g)')
            axes[i].grid(True, alpha=0.3)
            
            # Add zero line
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Gait Cycle Analysis - Subject {subject_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_multiple_cycles(self, dataset, subject_id, n_cycles=5, save_path=None):
        """
        Plot multiple gait cycles for a subject to show variability
        
        Args:
            dataset (dict): Dictionary containing gait data
            subject_id (str): Subject identifier
            n_cycles (int): Number of cycles to plot
            save_path (str): Path to save the plot
        """
        if subject_id not in dataset:
            print(f"Subject {subject_id} not found in dataset")
            return
        
        cycles = dataset[subject_id][:n_cycles]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        signals = ['vertical_acc', 'ap_acc', 'ml_acc']
        titles = ['Vertical Acceleration', 'Anterior-Posterior Acceleration', 'Medial-Lateral Acceleration']
        colors = ['red', 'blue', 'green']
        
        for i, (signal, title, color) in enumerate(zip(signals, titles, colors)):
            for j, cycle in enumerate(cycles):
                alpha = 0.7 - (j * 0.1)  # Decreasing opacity for overlay
                axes[i].plot(cycle['time'], cycle[signal], 
                           color=color, alpha=alpha, linewidth=1.5, label=f'Cycle {j+1}')
            
            axes[i].set_title(f'{title} - Multiple Cycles', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Time (normalized)')
            axes[i].set_ylabel('Acceleration (g)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.suptitle(f'Gait Variability Analysis - Subject {subject_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distribution(self, features_df, labels, feature_name, save_path=None):
        """
        Plot distribution of a specific feature across subjects
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            feature_name (str): Name of the feature to plot
            save_path (str): Path to save the plot
        """
        if feature_name not in features_df.columns:
            print(f"Feature {feature_name} not found")
            return
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'subject': labels,
            'feature_value': features_df[feature_name]
        })
        
        plt.figure(figsize=self.figsize)
        
        # Box plot
        plt.subplot(2, 1, 1)
        sns.boxplot(data=plot_data, x='subject', y='feature_value')
        plt.title(f'Distribution of {feature_name} by Subject', fontweight='bold')
        plt.xlabel('Subject ID')
        plt.ylabel(feature_name)
        plt.xticks(rotation=45)
        
        # Histogram
        plt.subplot(2, 1, 2)
        sns.histplot(data=plot_data, x='feature_value', hue='subject', alpha=0.7)
        plt.title(f'Histogram of {feature_name}', fontweight='bold')
        plt.xlabel(feature_value)
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_correlation_matrix(self, features_df, save_path=None):
        """
        Plot correlation matrix of features
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            save_path (str): Path to save the plot
        """
        # Select numeric columns only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_features.corr()
        
        # Create heatmap
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_visualization(self, features_df, labels, n_components=2, save_path=None):
        """
        Plot PCA visualization of features
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            n_components (int): Number of PCA components
            save_path (str): Path to save the plot
        """
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(features_clean)
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'subject': labels
        })
        
        if n_components >= 3:
            plot_data['PC3'] = pca_result[:, 2]
        
        plt.figure(figsize=self.figsize)
        
        if n_components == 2:
            # 2D scatter plot
            scatter = plt.scatter(plot_data['PC1'], plot_data['PC2'], 
                               c=plot_data['subject'], cmap='tab20', alpha=0.7)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, label='Subject ID')
            
        elif n_components >= 3:
            # 3D scatter plot
            ax = plt.axes(projection='3d')
            scatter = ax.scatter3D(plot_data['PC1'], plot_data['PC2'], plot_data['PC3'],
                                 c=plot_data['subject'], cmap='tab20', alpha=0.7)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
        
        plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_[:n_components].sum():.2%})', 
                 fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_visualization(self, features_df, labels, n_components=2, perplexity=30, save_path=None):
        """
        Plot t-SNE visualization of features
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            n_components (int): Number of t-SNE components
            perplexity (int): t-SNE perplexity parameter
            save_path (str): Path to save the plot
        """
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(features_clean)
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            't-SNE 1': tsne_result[:, 0],
            't-SNE 2': tsne_result[:, 1],
            'subject': labels
        })
        
        if n_components >= 3:
            plot_data['t-SNE 3'] = tsne_result[:, 2]
        
        plt.figure(figsize=self.figsize)
        
        if n_components == 2:
            scatter = plt.scatter(plot_data['t-SNE 1'], plot_data['t-SNE 2'], 
                               c=plot_data['subject'], cmap='tab20', alpha=0.7)
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.colorbar(scatter, label='Subject ID')
            
        elif n_components >= 3:
            ax = plt.axes(projection='3d')
            scatter = ax.scatter3D(plot_data['t-SNE 1'], plot_data['t-SNE 2'], plot_data['t-SNE 3'],
                                 c=plot_data['subject'], cmap='tab20', alpha=0.7)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
        
        plt.title('t-SNE Visualization of Gait Features', fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_gait_cycle(self, cycle_data, subject_id=None):
        """
        Create interactive plot of gait cycle using Plotly
        
        Args:
            cycle_data (dict): Dictionary containing gait cycle data
            subject_id (str): Subject identifier
        """
        time = cycle_data['time']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Vertical Acceleration', 'Anterior-Posterior Acceleration', 'Medial-Lateral Acceleration'),
            vertical_spacing=0.1
        )
        
        signals = ['vertical_acc', 'ap_acc', 'ml_acc']
        colors = ['red', 'blue', 'green']
        
        for i, (signal, color) in enumerate(zip(signals, colors)):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=cycle_data[signal],
                    mode='lines',
                    name=signal.replace('_', ' ').title(),
                    line=dict(color=color, width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=f'Interactive Gait Cycle Analysis - Subject {subject_id}',
            height=800,
            showlegend=True
        )
        
        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(title_text="Time (normalized)", row=i, col=1)
            fig.update_yaxes(title_text="Acceleration (g)", row=i, col=1)
        
        fig.show()
    
    def plot_model_performance_comparison(self, results, save_path=None):
        """
        Plot comparison of model performance
        
        Args:
            results (dict): Dictionary containing model results
            save_path (str): Path to save the plot
        """
        model_names = list(results.keys())
        accuracies = [results[name].get('accuracy', 0) for name in model_names]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, dataset, features_df, labels, results=None, save_dir=None):
        """
        Create a comprehensive visualization report
        
        Args:
            dataset (dict): Gait dataset
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            results (dict): Model results
            save_dir (str): Directory to save plots
        """
        print("Creating comprehensive visualization report...")
        
        # 1. Sample gait cycles
        print("1. Plotting sample gait cycles...")
        sample_subjects = list(dataset.keys())[:3]
        for subject_id in sample_subjects:
            if dataset[subject_id]:
                self.plot_gait_cycle(dataset[subject_id][0], subject_id, 
                                   save_path=f"{save_dir}/gait_cycle_{subject_id}.png" if save_dir else None)
        
        # 2. Feature correlation
        print("2. Plotting feature correlation matrix...")
        self.plot_feature_correlation_matrix(features_df, 
                                           save_path=f"{save_dir}/feature_correlation.png" if save_dir else None)
        
        # 3. PCA visualization
        print("3. Creating PCA visualization...")
        self.plot_pca_visualization(features_df, labels, 
                                  save_path=f"{save_dir}/pca_visualization.png" if save_dir else None)
        
        # 4. t-SNE visualization
        print("4. Creating t-SNE visualization...")
        self.plot_tsne_visualization(features_df, labels, 
                                   save_path=f"{save_dir}/tsne_visualization.png" if save_dir else None)
        
        # 5. Model performance (if available)
        if results:
            print("5. Plotting model performance...")
            self.plot_model_performance_comparison(results, 
                                                 save_path=f"{save_dir}/model_performance.png" if save_dir else None)
        
        print("Comprehensive report created successfully!")

def demo_visualization():
    """
    Demonstration of visualization capabilities
    """
    from data_preprocessing import create_sample_dataset
    from feature_extraction import create_feature_dataset
    
    print("=== Gait Visualization Demo ===\n")
    
    # Load sample data
    print("Loading sample data...")
    dataset = create_sample_dataset()
    features_df, labels = create_feature_dataset()
    
    # Initialize visualizer
    visualizer = GaitVisualizer()
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(dataset, features_df, labels)

if __name__ == "__main__":
    demo_visualization()
