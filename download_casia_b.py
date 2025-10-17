"""
CASIA-B Dataset Download and Setup Script
Handles downloading and setting up the official CASIA-B dataset
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class CASIABDownloader:
    """
    Handles CASIA-B dataset download and setup
    """
    
    def __init__(self, download_dir: str = "casia_b_dataset"):
        """
        Initialize CASIA-B downloader
        
        Args:
            download_dir (str): Directory to store the dataset
        """
        self.download_dir = download_dir
        self.dataset_path = os.path.join(download_dir, "GaitDatasetB-silh")
        self.zip_path = os.path.join(download_dir, "GaitDatasetB-silh.zip")
        
    def check_dataset_exists(self) -> bool:
        """
        Check if CASIA-B dataset already exists
        
        Returns:
            bool: True if dataset exists, False otherwise
        """
        if os.path.exists(self.dataset_path):
            print("✅ CASIA-B dataset already exists!")
            return True
        return False
    
    def print_download_instructions(self):
        """
        Print instructions for downloading CASIA-B dataset
        """
        print("📥 CASIA-B Dataset Download Instructions")
        print("=" * 50)
        print()
        print("The CASIA-B dataset is available for academic research.")
        print("Follow these steps to download:")
        print()
        print("1. 🌐 Visit the official CASIA website:")
        print("   http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip")
        print()
        print("2. 📝 Fill out the download request form:")
        print("   - Provide your academic affiliation")
        print("   - Agree to the terms of use")
        print("   - Specify research purpose")
        print()
        print("3. ⏳ Wait for approval (usually 1-2 business days)")
        print()
        print("4. 📦 Download the dataset (approximately 2.3 GB)")
        print()
        print("5. 🗂️  Extract to your project directory")
        print()
        print("Alternative sources:")
        print("• Papers with Code: https://paperswithcode.com/dataset/casia-b")
        print("• ResearchGate: Search 'CASIA-B gait dataset'")
        print()
        print("⚠️  Important Notes:")
        print("• Dataset is for non-commercial use only")
        print("• Must cite the original paper if used in research")
        print("• Commercial use requires separate licensing")
        print()
    
    def create_sample_casia_b_structure(self):
        """
        Create a sample CASIA-B directory structure for testing
        """
        print("🔧 Creating sample CASIA-B directory structure...")
        
        # Create directory structure
        os.makedirs(self.dataset_path, exist_ok=True)
        
        # Create condition directories
        conditions = ['nm', 'bg', 'cl']
        for condition in conditions:
            condition_path = os.path.join(self.dataset_path, condition)
            os.makedirs(condition_path, exist_ok=True)
            
            # Create a few sample subject directories
            for subject_id in range(1, 6):  # First 5 subjects
                subject_dir = os.path.join(condition_path, f"{subject_id:03d}")
                os.makedirs(subject_dir, exist_ok=True)
                
                # Create angle subdirectories
                angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
                for angle in angles:
                    angle_dir = os.path.join(subject_dir, angle)
                    os.makedirs(angle_dir, exist_ok=True)
                    
                    # Create a few sample silhouette images
                    for i in range(5):  # 5 sample frames
                        img_path = os.path.join(angle_dir, f"{i:03d}.png")
                        self._create_sample_silhouette(img_path)
        
        print(f"✅ Sample structure created at: {self.dataset_path}")
        print("📁 Directory structure:")
        print("   casia_b_dataset/GaitDatasetB-silh/")
        print("   ├── nm/ (normal walking)")
        print("   ├── bg/ (with bag)")
        print("   ├── cl/ (with coat)")
        print("   └── [subject_id]/[angle]/[frame].png")
    
    def _create_sample_silhouette(self, img_path: str):
        """
        Create a sample silhouette image
        
        Args:
            img_path (str): Path to save the image
        """
        # Create a simple silhouette
        img = np.zeros((128, 64), dtype=np.uint8)
        
        # Draw a simple human silhouette
        # Head
        cv2.circle(img, (32, 20), 8, 255, -1)
        
        # Torso
        cv2.rectangle(img, (24, 28), (40, 60), 255, -1)
        
        # Arms
        cv2.rectangle(img, (16, 32), (24, 48), 255, -1)
        cv2.rectangle(img, (40, 32), (48, 48), 255, -1)
        
        # Legs
        cv2.rectangle(img, (26, 60), (32, 100), 255, -1)
        cv2.rectangle(img, (32, 60), (38, 100), 255, -1)
        
        # Save image
        Image.fromarray(img).save(img_path)
    
    def analyze_dataset_structure(self) -> Dict:
        """
        Analyze the CASIA-B dataset structure
        
        Returns:
            dict: Dataset analysis results
        """
        if not os.path.exists(self.dataset_path):
            print("❌ CASIA-B dataset not found. Please download first.")
            return {}
        
        print("🔍 Analyzing CASIA-B dataset structure...")
        
        analysis = {
            'conditions': {},
            'total_subjects': 0,
            'total_sequences': 0,
            'total_frames': 0
        }
        
        conditions = ['nm', 'bg', 'cl']
        angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        
        for condition in conditions:
            condition_path = os.path.join(self.dataset_path, condition)
            if os.path.exists(condition_path):
                subjects = [d for d in os.listdir(condition_path) if d.isdigit()]
                analysis['conditions'][condition] = {
                    'subjects': len(subjects),
                    'angles': {},
                    'total_sequences': 0,
                    'total_frames': 0
                }
                
                for subject in subjects:
                    subject_path = os.path.join(condition_path, subject)
                    
                    for angle in angles:
                        angle_path = os.path.join(subject_path, angle)
                        if os.path.exists(angle_path):
                            frames = [f for f in os.listdir(angle_path) if f.endswith('.png')]
                            
                            if angle not in analysis['conditions'][condition]['angles']:
                                analysis['conditions'][condition]['angles'][angle] = 0
                            
                            analysis['conditions'][condition]['angles'][angle] += 1
                            analysis['conditions'][condition]['total_sequences'] += 1
                            analysis['conditions'][condition]['total_frames'] += len(frames)
                            analysis['total_frames'] += len(frames)
                
                analysis['total_sequences'] += analysis['conditions'][condition]['total_sequences']
        
        analysis['total_subjects'] = len(set().union(*[
            os.listdir(os.path.join(self.dataset_path, condition))
            for condition in conditions
            if os.path.exists(os.path.join(self.dataset_path, condition))
        ]))
        
        return analysis
    
    def print_dataset_summary(self, analysis: Dict):
        """
        Print dataset analysis summary
        
        Args:
            analysis (dict): Dataset analysis results
        """
        if not analysis:
            return
        
        print("\n📊 CASIA-B Dataset Summary")
        print("=" * 40)
        print(f"👥 Total Subjects: {analysis['total_subjects']}")
        print(f"🎬 Total Sequences: {analysis['total_sequences']}")
        print(f"🖼️  Total Frames: {analysis['total_frames']}")
        
        print("\n📁 By Condition:")
        for condition, data in analysis['conditions'].items():
            condition_name = {
                'nm': 'Normal Walking',
                'bg': 'With Bag',
                'cl': 'With Coat'
            }.get(condition, condition)
            
            print(f"  {condition_name} ({condition}):")
            print(f"    Subjects: {data['subjects']}")
            print(f"    Sequences: {data['total_sequences']}")
            print(f"    Frames: {data['total_frames']}")
        
        print("\n📐 By Viewing Angle:")
        angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        for angle in angles:
            total_sequences = sum(
                data['angles'].get(angle, 0)
                for data in analysis['conditions'].values()
            )
            print(f"  {angle}°: {total_sequences} sequences")
    
    def visualize_sample_sequences(self, n_samples: int = 3):
        """
        Visualize sample gait sequences from the dataset
        
        Args:
            n_samples (int): Number of sample sequences to visualize
        """
        if not os.path.exists(self.dataset_path):
            print("❌ CASIA-B dataset not found. Please download first.")
            return
        
        print(f"🖼️  Visualizing {n_samples} sample sequences...")
        
        conditions = ['nm', 'bg', 'cl']
        angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        
        fig, axes = plt.subplots(n_samples, 11, figsize=(20, n_samples * 2))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            condition = conditions[i % len(conditions)]
            subject_id = f"{(i % 5) + 1:03d}"
            
            for j, angle in enumerate(angles):
                sequence_path = os.path.join(self.dataset_path, condition, subject_id, angle)
                
                if os.path.exists(sequence_path):
                    frames = sorted([f for f in os.listdir(sequence_path) if f.endswith('.png')])
                    
                    if frames:
                        # Load middle frame
                        middle_frame_idx = len(frames) // 2
                        img_path = os.path.join(sequence_path, frames[middle_frame_idx])
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        axes[i, j].imshow(img, cmap='gray')
                        axes[i, j].set_title(f'{condition} {subject_id} {angle}°')
                        axes[i, j].axis('off')
                    else:
                        axes[i, j].text(0.5, 0.5, 'No frames', ha='center', va='center')
                        axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, 'Not found', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.suptitle('CASIA-B Dataset Sample Sequences', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def create_dataset_info_file(self):
        """
        Create a dataset information file
        """
        info_content = """
# CASIA-B Dataset Information

## Dataset Overview
- **Name**: CASIA-B Gait Dataset
- **Subjects**: 124 individuals
- **Conditions**: 3 (normal, with bag, with coat)
- **Viewing Angles**: 11 (0° to 180° in 18° increments)
- **Format**: Binary silhouette images
- **Total Size**: ~2.3 GB

## Directory Structure
```
GaitDatasetB-silh/
├── nm/           # Normal walking
│   ├── 001/      # Subject 001
│   │   ├── 000/  # 0° viewing angle
│   │   ├── 018/  # 18° viewing angle
│   │   └── ...
│   └── ...
├── bg/           # Walking with bag
└── cl/           # Walking with coat
```

## Usage Notes
- Dataset is for academic research only
- Non-commercial use permitted
- Commercial use requires separate licensing
- Must cite original paper when used in research

## Citation
If you use this dataset, please cite:
```
@article{tan2006gait,
  title={Gait recognition based on multiple views},
  author={Tan, Dang and Huang, Kaiqi and Yu, Shihong and Tan, Tieniu},
  journal={Pattern Recognition},
  year={2006}
}
```

## Contact
For dataset access: casia-gait@nlpr.ia.ac.cn
Official website: http://www.cbsr.ia.ac.cn/
        """
        
        info_path = os.path.join(self.download_dir, "dataset_info.md")
        with open(info_path, 'w') as f:
            f.write(info_content)
        
        print(f"📄 Dataset information saved to: {info_path}")

def main():
    """
    Main function to handle CASIA-B dataset setup
    """
    print("🎯 CASIA-B Dataset Setup")
    print("=" * 40)
    
    downloader = CASIABDownloader()
    
    # Check if dataset exists
    if downloader.check_dataset_exists():
        # Analyze existing dataset
        analysis = downloader.analyze_dataset_structure()
        downloader.print_dataset_summary(analysis)
        
        # Visualize samples
        downloader.visualize_sample_sequences(n_samples=2)
        
    else:
        # Print download instructions
        downloader.print_download_instructions()
        
        # Create sample structure for testing
        create_sample = input("\nCreate sample directory structure for testing? (y/n): ").lower()
        if create_sample == 'y':
            downloader.create_sample_casia_b_structure()
            downloader.create_dataset_info_file()
            
            # Analyze the sample structure
            analysis = downloader.analyze_dataset_structure()
            downloader.print_dataset_summary(analysis)
    
    print("\n✅ CASIA-B dataset setup complete!")

if __name__ == "__main__":
    main()
