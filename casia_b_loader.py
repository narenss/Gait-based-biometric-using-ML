"""
CASIA-B Dataset Loader for Gait-Based Biometric Recognition
Handles loading and preprocessing of CASIA-B gait dataset
"""

import os
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CASIABLoader:
    """
    Loader for CASIA-B gait dataset
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize CASIA-B loader
        
        Args:
            dataset_path (str): Path to CASIA-B dataset directory
        """
        self.dataset_path = dataset_path
        self.subjects = []
        self.sequences = {}
        
    def load_dataset_info(self):
        """
        Load dataset structure information
        """
        print("CASIA-B Dataset Structure:")
        print("- 124 subjects (001-124)")
        print("- 3 conditions: nm (normal), bg (bag), cl (coat)")
        print("- 11 viewing angles: 000, 018, 036, 054, 072, 090, 108, 126, 144, 162, 180")
        print("- Silhouette sequences (binary images)")
        print("- Typical sequence length: 60-120 frames")
        
    def create_synthetic_casia_b_data(self, n_subjects=30, n_sequences_per_subject=20):
        """
        Create synthetic CASIA-B style data for demonstration
        This simulates the CASIA-B dataset structure and characteristics
        
        Args:
            n_subjects (int): Number of subjects to generate
            n_sequences_per_subject (int): Number of sequences per subject
            
        Returns:
            dict: Dictionary containing synthetic CASIA-B data
        """
        print(f"Generating synthetic CASIA-B data: {n_subjects} subjects, {n_sequences_per_subject} sequences each")
        
        np.random.seed(42)
        dataset = {}
        
        # CASIA-B conditions
        conditions = ['nm', 'bg', 'cl']  # normal, bag, coat
        angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        
        for subject_id in range(1, n_subjects + 1):
            subject_key = f"{subject_id:03d}"
            dataset[subject_key] = {}
            
            for condition in conditions:
                dataset[subject_key][condition] = {}
                
                for angle in angles:
                    # Generate gait sequence for this subject/condition/angle
                    sequence = self._generate_gait_sequence(subject_id, condition, angle)
                    dataset[subject_key][condition][angle] = sequence
        
        return dataset
    
    def _generate_gait_sequence(self, subject_id: int, condition: str, angle: int) -> np.ndarray:
        """
        Generate a synthetic gait sequence for a specific subject/condition/angle
        
        Args:
            subject_id (int): Subject identifier
            condition (str): Walking condition (nm/bg/cl)
            angle (int): Viewing angle
            
        Returns:
            np.ndarray: Gait sequence (frames, height, width)
        """
        # Sequence parameters
        n_frames = np.random.randint(60, 120)  # Variable sequence length
        height, width = 128, 64  # CASIA-B silhouette dimensions
        
        # Subject-specific characteristics
        seed = (subject_id * 1000 + hash(condition) + angle) % (2**32 - 1)
        np.random.seed(seed)
        subject_height = np.random.normal(1.0, 0.1)  # Height variation
        subject_width = np.random.normal(1.0, 0.08)  # Width variation
        gait_speed = np.random.normal(1.0, 0.15)  # Walking speed variation
        
        # Condition-specific modifications
        if condition == 'bg':  # Carrying bag
            subject_width *= 1.2
        elif condition == 'cl':  # Wearing coat
            subject_width *= 1.3
            subject_height *= 1.05
        
        # Generate gait cycle
        sequence = []
        for frame in range(n_frames):
            # Gait cycle position (0 to 2π)
            cycle_pos = (frame * gait_speed * 0.1) % (2 * np.pi)
            
            # Generate silhouette for this frame
            silhouette = self._generate_silhouette_frame(
                cycle_pos, subject_height, subject_width, angle, height, width
            )
            sequence.append(silhouette)
        
        return np.array(sequence)
    
    def _generate_silhouette_frame(self, cycle_pos: float, subject_height: float, 
                                 subject_width: float, angle: int, height: int, width: int) -> np.ndarray:
        """
        Generate a single silhouette frame
        
        Args:
            cycle_pos (float): Position in gait cycle (0 to 2π)
            subject_height (float): Subject height factor
            subject_width (float): Subject width factor
            angle (int): Viewing angle
            height (int): Frame height
            width (int): Frame width
            
        Returns:
            np.ndarray: Binary silhouette image
        """
        # Create empty frame
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate body position based on gait cycle
        x_center = width // 2 + int(10 * np.sin(cycle_pos))  # Lateral movement
        y_center = height // 2 + int(5 * np.cos(cycle_pos * 0.5))  # Vertical movement
        
        # Generate body parts based on viewing angle
        if angle <= 90:  # Side view
            # Head
            head_radius = int(8 * subject_height)
            cv2.circle(frame, (x_center, y_center - int(25 * subject_height)), 
                      head_radius, 255, -1)
            
            # Torso
            torso_width = int(12 * subject_width)
            torso_height = int(30 * subject_height)
            cv2.rectangle(frame, 
                         (x_center - torso_width//2, y_center - int(15 * subject_height)),
                         (x_center + torso_width//2, y_center + int(15 * subject_height)),
                         255, -1)
            
            # Legs (with gait cycle)
            leg_separation = int(8 * np.sin(cycle_pos))
            leg_width = int(4 * subject_width)
            leg_height = int(25 * subject_height)
            
            # Left leg
            cv2.rectangle(frame,
                         (x_center - torso_width//2 - leg_separation, y_center + int(15 * subject_height)),
                         (x_center - torso_width//2 + leg_width - leg_separation, y_center + int(40 * subject_height)),
                         255, -1)
            
            # Right leg
            cv2.rectangle(frame,
                         (x_center + torso_width//2 - leg_width + leg_separation, y_center + int(15 * subject_height)),
                         (x_center + torso_width//2 + leg_separation, y_center + int(40 * subject_height)),
                         255, -1)
            
            # Arms
            arm_swing = int(10 * np.sin(cycle_pos + np.pi/4))
            arm_width = int(3 * subject_width)
            arm_height = int(15 * subject_height)
            
            # Left arm
            cv2.rectangle(frame,
                         (x_center - torso_width//2 - arm_width - arm_swing, y_center - int(10 * subject_height)),
                         (x_center - torso_width//2 - arm_swing, y_center + int(5 * subject_height)),
                         255, -1)
            
            # Right arm
            cv2.rectangle(frame,
                         (x_center + torso_width//2 + arm_swing, y_center - int(10 * subject_height)),
                         (x_center + torso_width//2 + arm_width + arm_swing, y_center + int(5 * subject_height)),
                         255, -1)
        
        else:  # Front/back view
            # Simplified front view representation
            body_width = int(15 * subject_width)
            body_height = int(40 * subject_height)
            
            cv2.rectangle(frame,
                         (x_center - body_width//2, y_center - body_height//2),
                         (x_center + body_width//2, y_center + body_height//2),
                         255, -1)
        
        # Apply some noise to make it more realistic
        noise = np.random.random((height, width)) < 0.02
        frame[noise] = 255 - frame[noise]
        
        return frame
    
    def extract_gait_features_from_sequence(self, sequence: np.ndarray) -> Dict:
        """
        Extract gait features from a silhouette sequence
        
        Args:
            sequence (np.ndarray): Gait sequence (frames, height, width)
            
        Returns:
            dict: Extracted gait features
        """
        features = {}
        
        # Temporal features
        features['sequence_length'] = len(sequence)
        features['avg_silhouette_area'] = np.mean([np.sum(frame > 0) for frame in sequence])
        features['silhouette_area_std'] = np.std([np.sum(frame > 0) for frame in sequence])
        
        # Spatial features
        centroids = []
        for frame in sequence:
            if np.sum(frame > 0) > 0:
                moments = cv2.moments(frame)
                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    centroids.append([cx, cy])
        
        if len(centroids) > 1:
            centroids = np.array(centroids)
            features['center_of_mass_variation_x'] = np.std(centroids[:, 0])
            features['center_of_mass_variation_y'] = np.std(centroids[:, 1])
            features['center_of_mass_range_x'] = np.max(centroids[:, 0]) - np.min(centroids[:, 0])
            features['center_of_mass_range_y'] = np.max(centroids[:, 1]) - np.min(centroids[:, 1])
        else:
            features.update({
                'center_of_mass_variation_x': 0,
                'center_of_mass_variation_y': 0,
                'center_of_mass_range_x': 0,
                'center_of_mass_range_y': 0
            })
        
        # Gait cycle features
        silhouette_areas = [np.sum(frame > 0) for frame in sequence]
        if len(silhouette_areas) > 1:
            # Find peaks and valleys for gait cycle analysis
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(silhouette_areas, height=np.mean(silhouette_areas))
            valleys, _ = find_peaks([-x for x in silhouette_areas], height=-np.mean(silhouette_areas))
            
            features['gait_cycle_peaks'] = len(peaks)
            features['gait_cycle_valleys'] = len(valleys)
            features['gait_regularity'] = 1 / (1 + np.std(silhouette_areas) / (np.mean(silhouette_areas) + 1e-8))
        else:
            features.update({
                'gait_cycle_peaks': 0,
                'gait_cycle_valleys': 0,
                'gait_regularity': 0
            })
        
        # Width and height variations
        widths = []
        heights = []
        for frame in sequence:
            if np.sum(frame > 0) > 0:
                coords = np.where(frame > 0)
                if len(coords[0]) > 0:
                    widths.append(np.max(coords[1]) - np.min(coords[1]))
                    heights.append(np.max(coords[0]) - np.min(coords[0]))
        
        if widths and heights:
            features['width_variation'] = np.std(widths) / (np.mean(widths) + 1e-8)
            features['height_variation'] = np.std(heights) / (np.mean(heights) + 1e-8)
            features['avg_width'] = np.mean(widths)
            features['avg_height'] = np.mean(heights)
        else:
            features.update({
                'width_variation': 0,
                'height_variation': 0,
                'avg_width': 0,
                'avg_height': 0
            })
        
        return features
    
    def create_feature_dataset_from_casia_b(self, dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature dataset from CASIA-B data
        
        Args:
            dataset (dict): CASIA-B dataset
            
        Returns:
            tuple: (features_df, labels) where features_df is DataFrame of features and labels is array of subject IDs
        """
        all_features = []
        labels = []
        
        print("Extracting features from CASIA-B data...")
        
        for subject_id, subject_data in dataset.items():
            subject_label = int(subject_id)
            
            for condition, condition_data in subject_data.items():
                for angle, sequence in condition_data.items():
                    # Extract features from this sequence
                    features = self.extract_gait_features_from_sequence(sequence)
                    
                    # Add metadata features
                    features['subject_id'] = subject_label
                    features['condition'] = condition
                    features['angle'] = angle
                    
                    all_features.append(features)
                    labels.append(subject_label)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        print(f"Extracted {features_df.shape[1]} features from {len(labels)} sequences")
        print(f"Number of unique subjects: {len(np.unique(labels))}")
        
        return features_df, np.array(labels)
    
    def visualize_gait_sequence(self, sequence: np.ndarray, title: str = "Gait Sequence"):
        """
        Visualize a gait sequence
        
        Args:
            sequence (np.ndarray): Gait sequence
            title: str): Plot title
        """
        n_frames = len(sequence)
        n_cols = min(5, n_frames)
        n_rows = min(2, (n_frames + n_cols - 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(min(10, n_frames)):
            frame_idx = i * (n_frames - 1) // 9 if n_frames > 1 else 0
            axes[i].imshow(sequence[frame_idx], cmap='gray')
            axes[i].set_title(f'Frame {frame_idx}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(min(10, n_frames), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def create_casia_b_demo():
    """
    Create a demonstration using synthetic CASIA-B data
    """
    print("=== CASIA-B Dataset Integration Demo ===\n")
    
    # Initialize loader
    loader = CASIABLoader()
    loader.load_dataset_info()
    
    # Generate synthetic CASIA-B data
    print("\nGenerating synthetic CASIA-B data...")
    dataset = loader.create_synthetic_casia_b_data(n_subjects=20, n_sequences_per_subject=10)
    
    # Extract features
    features_df, labels = loader.create_feature_dataset_from_casia_b(dataset)
    
    # Visualize sample sequence
    sample_subject = list(dataset.keys())[0]
    sample_sequence = dataset[sample_subject]['nm'][90]  # Normal condition, side view
    loader.visualize_gait_sequence(sample_sequence, f"Sample Gait Sequence - Subject {sample_subject}")
    
    return dataset, features_df, labels

if __name__ == "__main__":
    # Run CASIA-B demo
    dataset, features_df, labels = create_casia_b_demo()
    
    print(f"\nDataset Summary:")
    print(f"- Subjects: {len(dataset)}")
    print(f"- Features: {features_df.shape[1]}")
    print(f"- Sequences: {len(labels)}")
    print(f"- Unique subjects: {len(np.unique(labels))}")
