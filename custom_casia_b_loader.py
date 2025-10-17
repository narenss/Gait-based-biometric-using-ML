"""
Custom CASIA-B Loader for Your Dataset Format
Handles your specific CASIA-B directory structure
"""

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class CustomCASIABLoader:
    """
    Custom loader for your CASIA-B dataset format
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize loader for your CASIA-B dataset
        
        Args:
            dataset_path (str): Path to CASIA-B dataset
        """
        self.dataset_path = dataset_path
        self.sequences = {}
        self.features_df = None
        self.labels = None
        
        # Your dataset structure
        self.conditions = ['nm', 'bg', 'cl']  # normal, bag, coat
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        
    def load_gait_sequences(self, max_subjects: int = None, max_sequences_per_subject: int = 3):
        """
        Load gait sequences from your CASIA-B dataset
        
        Args:
            max_subjects (int): Maximum number of subjects to load
            max_sequences_per_subject (int): Maximum sequences per subject
        """
        print("📂 Loading gait sequences from your CASIA-B dataset...")
        
        self.sequences = {}
        total_loaded = 0
        
        # Get all subject directories
        subjects = [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()]
        
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        print(f"Found {len(subjects)} subjects")
        
        for subject_id in subjects:
            subject_path = os.path.join(self.dataset_path, subject_id)
            print(f"  Processing subject {subject_id}...")
            
            self.sequences[subject_id] = {}
            sequences_loaded = 0
            
            # Look for condition directories (nm-01, bg-01, etc.)
            for item in os.listdir(subject_path):
                if os.path.isdir(os.path.join(subject_path, item)):
                    # Extract condition from directory name
                    if item.startswith('nm-'):
                        condition = 'nm'
                    elif item.startswith('bg-'):
                        condition = 'bg'
                    elif item.startswith('cl-'):
                        condition = 'cl'
                    else:
                        continue
                    
                    if condition not in self.sequences[subject_id]:
                        self.sequences[subject_id][condition] = {}
                    
                    condition_path = os.path.join(subject_path, item)
                    
                    # Load sequences for each angle
                    for angle in self.angles:
                        angle_path = os.path.join(condition_path, angle)
                        if os.path.exists(angle_path):
                            sequence = self._load_sequence_from_directory(angle_path)
                            if sequence is not None and len(sequence) > 0:
                                self.sequences[subject_id][condition][angle] = sequence
                                total_loaded += 1
                                sequences_loaded += 1
                                
                                # Limit sequences per subject
                                if sequences_loaded >= max_sequences_per_subject * len(self.angles):
                                    break
                    
                    # Limit sequences per subject
                    if sequences_loaded >= max_sequences_per_subject * len(self.angles):
                        break
        
        print(f"✅ Loaded {total_loaded} gait sequences")
        return total_loaded
    
    def _load_sequence_from_directory(self, directory_path: str) -> np.ndarray:
        """
        Load a gait sequence from directory containing silhouette images
        
        Args:
            directory_path (str): Path to directory containing images
            
        Returns:
            np.ndarray: Array of silhouette images
        """
        try:
            # Get all PNG files
            image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
            image_files.sort()  # Sort to maintain temporal order
            
            if len(image_files) == 0:
                return None
            
            # Load images
            sequence = []
            for img_file in image_files:
                img_path = os.path.join(directory_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to standard size if needed
                    img = cv2.resize(img, (64, 128))
                    sequence.append(img)
            
            return np.array(sequence) if sequence else None
            
        except Exception as e:
            print(f"⚠️  Error loading sequence from {directory_path}: {str(e)}")
            return None
    
    def extract_gait_features(self, sequence: np.ndarray) -> dict:
        """
        Extract comprehensive features from gait sequence
        
        Args:
            sequence (np.ndarray): Gait sequence (frames, height, width)
            
        Returns:
            dict: Extracted features
        """
        features = {}
        
        if sequence is None or len(sequence) == 0:
            return self._get_default_features()
        
        # Basic sequence features
        features['sequence_length'] = len(sequence)
        features['avg_silhouette_area'] = np.mean([np.sum(frame > 0) for frame in sequence])
        features['silhouette_area_std'] = np.std([np.sum(frame > 0) for frame in sequence])
        features['silhouette_area_range'] = np.max([np.sum(frame > 0) for frame in sequence]) - np.min([np.sum(frame > 0) for frame in sequence])
        
        # Centroid analysis
        centroids = self._calculate_centroids(sequence)
        if len(centroids) > 1:
            centroids = np.array(centroids)
            features.update({
                'center_of_mass_variation_x': np.std(centroids[:, 0]),
                'center_of_mass_variation_y': np.std(centroids[:, 1]),
                'center_of_mass_range_x': np.max(centroids[:, 0]) - np.min(centroids[:, 0]),
                'center_of_mass_range_y': np.max(centroids[:, 1]) - np.min(centroids[:, 1]),
                'center_of_mass_velocity_x': np.mean(np.abs(np.diff(centroids[:, 0]))),
                'center_of_mass_velocity_y': np.mean(np.abs(np.diff(centroids[:, 1]))),
            })
        else:
            features.update({
                'center_of_mass_variation_x': 0, 'center_of_mass_variation_y': 0,
                'center_of_mass_range_x': 0, 'center_of_mass_range_y': 0,
                'center_of_mass_velocity_x': 0, 'center_of_mass_velocity_y': 0,
            })
        
        # Gait cycle analysis
        gait_features = self._analyze_gait_cycle(sequence)
        features.update(gait_features)
        
        # Shape features
        shape_features = self._extract_shape_features(sequence)
        features.update(shape_features)
        
        # Temporal dynamics
        temporal_features = self._extract_temporal_dynamics(sequence)
        features.update(temporal_features)
        
        return features
    
    def _calculate_centroids(self, sequence: np.ndarray) -> list:
        """Calculate centroids for each frame"""
        centroids = []
        for frame in sequence:
            if np.sum(frame > 0) > 0:
                moments = cv2.moments(frame)
                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    centroids.append([cx, cy])
        return centroids
    
    def _analyze_gait_cycle(self, sequence: np.ndarray) -> dict:
        """Analyze gait cycle characteristics"""
        features = {}
        
        silhouette_areas = [np.sum(frame > 0) for frame in sequence]
        
        if len(silhouette_areas) > 1:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(silhouette_areas, height=np.mean(silhouette_areas))
            valleys, _ = find_peaks([-x for x in silhouette_areas], height=-np.mean(silhouette_areas))
            
            features.update({
                'gait_cycle_peaks': len(peaks),
                'gait_cycle_valleys': len(valleys),
                'gait_regularity': 1 / (1 + np.std(silhouette_areas) / (np.mean(silhouette_areas) + 1e-8)),
                'gait_cycle_frequency': len(peaks) / len(sequence),
                'gait_symmetry': self._calculate_gait_symmetry(silhouette_areas),
                'gait_rhythm': self._calculate_gait_rhythm(silhouette_areas),
            })
        else:
            features.update({
                'gait_cycle_peaks': 0, 'gait_cycle_valleys': 0, 'gait_regularity': 0,
                'gait_cycle_frequency': 0, 'gait_symmetry': 0, 'gait_rhythm': 0,
            })
        
        return features
    
    def _extract_shape_features(self, sequence: np.ndarray) -> dict:
        """Extract shape and proportion features"""
        features = {}
        
        widths, heights, aspect_ratios = [], [], []
        for frame in sequence:
            if np.sum(frame > 0) > 0:
                coords = np.where(frame > 0)
                if len(coords[0]) > 0:
                    width = np.max(coords[1]) - np.min(coords[1])
                    height = np.max(coords[0]) - np.min(coords[0])
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / (height + 1e-8))
        
        if widths and heights:
            features.update({
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'width_variation': np.std(widths) / (np.mean(widths) + 1e-8),
                'height_variation': np.std(heights) / (np.mean(heights) + 1e-8),
                'avg_aspect_ratio': np.mean(aspect_ratios),
                'aspect_ratio_variation': np.std(aspect_ratios) / (np.mean(aspect_ratios) + 1e-8),
            })
        else:
            features.update({
                'avg_width': 0, 'avg_height': 0, 'width_variation': 0, 'height_variation': 0,
                'avg_aspect_ratio': 0, 'aspect_ratio_variation': 0,
            })
        
        return features
    
    def _extract_temporal_dynamics(self, sequence: np.ndarray) -> dict:
        """Extract temporal dynamics features"""
        features = {}
        
        # Calculate frame-to-frame differences
        frame_diffs = []
        for i in range(1, len(sequence)):
            diff = np.sum(np.abs(sequence[i] - sequence[i-1]))
            frame_diffs.append(diff)
        
        if frame_diffs:
            features.update({
                'avg_frame_difference': np.mean(frame_diffs),
                'frame_difference_std': np.std(frame_diffs),
                'temporal_smoothness': 1 / (1 + np.std(frame_diffs) / (np.mean(frame_diffs) + 1e-8)),
            })
        else:
            features.update({
                'avg_frame_difference': 0, 'frame_difference_std': 0,
                'temporal_smoothness': 0,
            })
        
        return features
    
    def _calculate_gait_symmetry(self, areas: list) -> float:
        """Calculate gait symmetry"""
        if len(areas) < 2:
            return 0
        mid = len(areas) // 2
        first_half = areas[:mid]
        second_half = areas[mid:]
        return 1 - abs(np.mean(first_half) - np.mean(second_half)) / (np.mean(first_half) + np.mean(second_half) + 1e-8)
    
    def _calculate_gait_rhythm(self, areas: list) -> float:
        """Calculate gait rhythm consistency"""
        if len(areas) < 3:
            return 0
        intervals = np.diff(areas)
        return 1 / (1 + np.std(intervals) / (np.mean(np.abs(intervals)) + 1e-8))
    
    def _get_default_features(self) -> dict:
        """Get default features for empty sequences"""
        return {
            'sequence_length': 0, 'avg_silhouette_area': 0, 'silhouette_area_std': 0,
            'silhouette_area_range': 0, 'center_of_mass_variation_x': 0, 'center_of_mass_variation_y': 0,
            'center_of_mass_range_x': 0, 'center_of_mass_range_y': 0, 'center_of_mass_velocity_x': 0,
            'center_of_mass_velocity_y': 0, 'gait_cycle_peaks': 0, 'gait_cycle_valleys': 0,
            'gait_regularity': 0, 'gait_cycle_frequency': 0, 'gait_symmetry': 0, 'gait_rhythm': 0,
            'avg_width': 0, 'avg_height': 0, 'width_variation': 0, 'height_variation': 0,
            'avg_aspect_ratio': 0, 'aspect_ratio_variation': 0, 'avg_frame_difference': 0,
            'frame_difference_std': 0, 'temporal_smoothness': 0
        }
    
    def create_feature_dataset(self) -> tuple:
        """
        Create feature dataset from loaded sequences
        
        Returns:
            tuple: (features_df, labels)
        """
        print("🔍 Extracting features from CASIA-B sequences...")
        
        all_features = []
        labels = []
        
        for subject_id, subject_data in self.sequences.items():
            subject_label = int(subject_id)
            
            for condition, condition_data in subject_data.items():
                for angle, sequence in condition_data.items():
                    # Extract features
                    features = self.extract_gait_features(sequence)
                    
                    # Add metadata
                    features['subject_id'] = subject_label
                    features['condition'] = condition
                    features['angle'] = int(angle)
                    
                    all_features.append(features)
                    labels.append(subject_label)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(all_features)
        self.labels = np.array(labels)
        
        print(f"✅ Feature extraction complete:")
        print(f"   📊 Features: {self.features_df.shape[1]}")
        print(f"   🎯 Sequences: {len(labels)}")
        print(f"   👥 Subjects: {len(np.unique(labels))}")
        
        return self.features_df, self.labels

class HighAccuracyClassifier:
    """
    High-accuracy classifier for CASIA-B dataset
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        
    def prepare_data(self, features_df, labels, test_size=0.2):
        """Prepare data for training"""
        print("🔧 Preparing data for high-accuracy training...")
        
        # Remove metadata columns
        feature_columns = [col for col in features_df.columns 
                          if col not in ['subject_id', 'condition', 'angle']]
        X = features_df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            print(f"   Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=self.random_state, 
            stratify=labels, shuffle=True
        )
        
        # Feature selection
        k = min(25, X_train.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        print(f"✅ Data preparation complete:")
        print(f"   📊 Features: {X_train_scaled.shape[1]}")
        print(f"   📈 Training: {X_train_scaled.shape}")
        print(f"   📉 Testing: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def initialize_models(self):
        """Initialize high-accuracy models"""
        print("🚀 Initializing high-accuracy models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                min_samples_split=5, random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf', C=10.0, gamma='scale',
                random_state=self.random_state, probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=3, weights='distance'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(200, 100), activation='relu',
                solver='adam', alpha=0.001, max_iter=1000,
                random_state=self.random_state, early_stopping=True
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=2000, random_state=self.random_state,
                solver='lbfgs', multi_class='multinomial'
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("🎯 Training models...")
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                print(f"   ✅ {name}")
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("📊 Evaluating models...")
        
        results = {}
        best_accuracy = 0
        best_model_name = None
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'model': model
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    self.best_model = model
                
                status = "🎯 EXCELLENT" if accuracy >= 0.90 else "✅ GOOD" if accuracy >= 0.80 else "⚠️ MODERATE"
                print(f"   {name:<20}: {accuracy:.4f} {status}")
                
            except Exception as e:
                print(f"   {name:<20}: ERROR - {str(e)}")
                results[name] = {'error': str(e)}
        
        print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
        return results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """Perform cross-validation"""
        print(f"🔄 Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
                
                status = "🎯 EXCELLENT" if scores.mean() >= 0.90 else "✅ GOOD" if scores.mean() >= 0.80 else "⚠️ MODERATE"
                print(f"   {name:<20}: {scores.mean():.4f} ± {scores.std():.4f} {status}")
                
            except Exception as e:
                print(f"   {name:<20}: CV ERROR - {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results

def run_casia_b_training(dataset_path: str):
    """
    Run complete CASIA-B training experiment
    
    Args:
        dataset_path (str): Path to CASIA-B dataset
    """
    print("🎯 CASIA-B GAIT RECOGNITION TRAINING")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print("Goal: Achieve 90%+ accuracy for person identification")
    print()
    
    # Initialize processor
    processor = CustomCASIABLoader(dataset_path)
    
    # Load sequences (start with fewer subjects for testing)
    total_sequences = processor.load_gait_sequences(max_subjects=30, max_sequences_per_subject=2)
    if total_sequences == 0:
        print("❌ No sequences loaded. Please check the dataset.")
        return None
    
    # Extract features
    features_df, labels = processor.create_feature_dataset()
    
    # Initialize classifier
    classifier = HighAccuracyClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df, labels)
    
    # Initialize and train models
    classifier.initialize_models()
    classifier.train_models(X_train, y_train)
    
    # Cross-validation
    print("\n" + "="*60)
    cv_results = classifier.cross_validate_models(X_train, y_train)
    
    # Final evaluation
    print("\n" + "="*60)
    final_results = classifier.evaluate_models(X_test, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("🎯 EXPERIMENT SUMMARY")
    print("="*60)
    
    best_cv_score = 0
    best_cv_model = None
    for name, result in cv_results.items():
        if 'mean_score' in result and result['mean_score'] > best_cv_score:
            best_cv_score = result['mean_score']
            best_cv_model = name
    
    best_test_score = 0
    best_test_model = None
    for name, result in final_results.items():
        if 'accuracy' in result and result['accuracy'] > best_test_score:
            best_test_score = result['accuracy']
            best_test_model = name
    
    print(f"🏆 Best Cross-Validation Model: {best_cv_model} ({best_cv_score:.4f})")
    print(f"🏆 Best Test Model: {best_test_model} ({best_test_score:.4f})")
    
    # Goal achievement
    if best_test_score >= 0.90:
        print("🎉 GOAL ACHIEVED! 90%+ accuracy reached!")
        print("✅ The model can reliably identify who is walking!")
    elif best_test_score >= 0.80:
        print("✅ Good performance achieved!")
        print("💡 Close to 90% target - consider more data or feature engineering")
    else:
        print("⚠️  Performance below target")
        print("💡 Consider: more data, better features, or model tuning")
    
    print("\n🎯 What the model predicts:")
    print("   • Input: Gait sequence (video frames of walking)")
    print("   • Output: Person identity (Subject_001, Subject_087, etc.)")
    print("   • Accuracy: How often the model correctly identifies the person")
    
    return classifier, final_results, cv_results

if __name__ == "__main__":
    # Run with your dataset
    dataset_path = "./casia-b"
    classifier, results, cv_results = run_casia_b_training(dataset_path)
