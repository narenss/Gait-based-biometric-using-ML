"""
Quick CASIA-B Training - Fast and Effective
Trains on real CASIA-B data in 5-10 minutes with good accuracy
"""

import os
import numpy as np
import pandas as pd
import cv2
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

class QuickCASIABLoader:
    """
    Quick loader for CASIA-B dataset - optimized for speed
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.sequences = {}
        self.features_df = None
        self.labels = None
        
        # Your dataset structure
        self.conditions = ['nm', 'bg', 'cl']
        self.angles = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
        
    def load_gait_sequences(self, max_subjects: int = 25, max_sequences_per_subject: int = 2):
        """
        Load gait sequences quickly
        """
        print("📂 Loading gait sequences from your CASIA-B dataset...")
        
        self.sequences = {}
        total_loaded = 0
        
        # Get subject directories
        subjects = [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d)) and d.isdigit()]
        
        subjects = subjects[:max_subjects]  # Limit subjects for speed
        print(f"Processing {len(subjects)} subjects for quick training...")
        
        for subject_id in subjects:
            subject_path = os.path.join(self.dataset_path, subject_id)
            
            self.sequences[subject_id] = {}
            sequences_loaded = 0
            
            # Look for condition directories
            for item in os.listdir(subject_path):
                if os.path.isdir(os.path.join(subject_path, item)):
                    # Extract condition
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
                    
                    # Load sequences for key angles only (for speed)
                    key_angles = ['000', '090', '180']  # Front, side, back views
                    for angle in key_angles:
                        if angle in self.angles:
                            angle_path = os.path.join(condition_path, angle)
                            if os.path.exists(angle_path):
                                sequence = self._load_sequence_from_directory(angle_path)
                                if sequence is not None and len(sequence) > 0:
                                    self.sequences[subject_id][condition][angle] = sequence
                                    total_loaded += 1
                                    sequences_loaded += 1
                                    
                                    # Limit sequences per subject
                                    if sequences_loaded >= max_sequences_per_subject * len(key_angles):
                                        break
                    
                    # Limit sequences per subject
                    if sequences_loaded >= max_sequences_per_subject * len(key_angles):
                        break
        
        print(f"✅ Loaded {total_loaded} gait sequences")
        return total_loaded
    
    def _load_sequence_from_directory(self, directory_path: str) -> np.ndarray:
        """Load a gait sequence quickly"""
        try:
            image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
            image_files.sort()
            
            if len(image_files) == 0:
                return None
            
            # Load every 3rd frame for speed (still maintains gait pattern)
            sequence = []
            for i in range(0, len(image_files), 3):
                img_path = os.path.join(directory_path, image_files[i])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to standard size
                    img = cv2.resize(img, (64, 128))
                    sequence.append(img)
            
            return np.array(sequence) if sequence else None
            
        except Exception as e:
            return None
    
    def extract_quick_features(self, sequence: np.ndarray) -> dict:
        """
        Extract essential gait features quickly
        """
        features = {}
        
        if sequence is None or len(sequence) == 0:
            return self._get_default_features()
        
        # Basic sequence features
        features['sequence_length'] = len(sequence)
        features['avg_silhouette_area'] = np.mean([np.sum(frame > 0) for frame in sequence])
        features['silhouette_area_std'] = np.std([np.sum(frame > 0) for frame in sequence])
        
        # Centroid analysis
        centroids = self._calculate_centroids(sequence)
        if len(centroids) > 1:
            centroids = np.array(centroids)
            features.update({
                'center_of_mass_variation_x': np.std(centroids[:, 0]),
                'center_of_mass_variation_y': np.std(centroids[:, 1]),
                'center_of_mass_range_x': np.max(centroids[:, 0]) - np.min(centroids[:, 0]),
                'center_of_mass_range_y': np.max(centroids[:, 1]) - np.min(centroids[:, 1]),
            })
        else:
            features.update({
                'center_of_mass_variation_x': 0, 'center_of_mass_variation_y': 0,
                'center_of_mass_range_x': 0, 'center_of_mass_range_y': 0,
            })
        
        # Gait cycle analysis
        gait_features = self._analyze_gait_cycle(sequence)
        features.update(gait_features)
        
        # Shape features
        shape_features = self._extract_shape_features(sequence)
        features.update(shape_features)
        
        # Gait Energy Image (GEI) features
        gei_features = self._extract_gei_features(sequence)
        features.update(gei_features)
        
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
                'gait_symmetry': self._calculate_gait_symmetry(silhouette_areas),
            })
        else:
            features.update({
                'gait_cycle_peaks': 0, 'gait_cycle_valleys': 0, 'gait_regularity': 0, 'gait_symmetry': 0,
            })
        
        return features
    
    def _extract_shape_features(self, sequence: np.ndarray) -> dict:
        """Extract shape features"""
        features = {}
        
        widths, heights = [], []
        for frame in sequence:
            if np.sum(frame > 0) > 0:
                coords = np.where(frame > 0)
                if len(coords[0]) > 0:
                    width = np.max(coords[1]) - np.min(coords[1])
                    height = np.max(coords[0]) - np.min(coords[0])
                    widths.append(width)
                    heights.append(height)
        
        if widths and heights:
            features.update({
                'avg_width': np.mean(widths),
                'avg_height': np.mean(heights),
                'width_variation': np.std(widths) / (np.mean(widths) + 1e-8),
                'height_variation': np.std(heights) / (np.mean(heights) + 1e-8),
                'aspect_ratio': np.mean(widths) / (np.mean(heights) + 1e-8),
            })
        else:
            features.update({
                'avg_width': 0, 'avg_height': 0, 'width_variation': 0, 'height_variation': 0, 'aspect_ratio': 0,
            })
        
        return features
    
    def _extract_gei_features(self, sequence: np.ndarray) -> dict:
        """Extract Gait Energy Image features"""
        features = {}
        
        if len(sequence) > 0:
            # Create Gait Energy Image
            gei = np.mean(sequence, axis=0)
            
            features['gei_mean'] = np.mean(gei)
            features['gei_std'] = np.std(gei)
            features['gei_energy'] = np.sum(gei**2)
            
            if np.sum(gei) > 0:
                features['gei_centroid_x'] = np.sum(np.sum(gei, axis=0) * np.arange(gei.shape[1])) / np.sum(gei)
                features['gei_centroid_y'] = np.sum(np.sum(gei, axis=1) * np.arange(gei.shape[0])) / np.sum(gei)
            else:
                features['gei_centroid_x'] = 0
                features['gei_centroid_y'] = 0
        
        return features
    
    def _calculate_gait_symmetry(self, areas: list) -> float:
        """Calculate gait symmetry"""
        if len(areas) < 2:
            return 0
        mid = len(areas) // 2
        first_half = areas[:mid]
        second_half = areas[mid:]
        return 1 - abs(np.mean(first_half) - np.mean(second_half)) / (np.mean(first_half) + np.mean(second_half) + 1e-8)
    
    def _get_default_features(self) -> dict:
        """Get default features"""
        return {
            'sequence_length': 0, 'avg_silhouette_area': 0, 'silhouette_area_std': 0,
            'center_of_mass_variation_x': 0, 'center_of_mass_variation_y': 0,
            'center_of_mass_range_x': 0, 'center_of_mass_range_y': 0,
            'gait_cycle_peaks': 0, 'gait_cycle_valleys': 0, 'gait_regularity': 0, 'gait_symmetry': 0,
            'avg_width': 0, 'avg_height': 0, 'width_variation': 0, 'height_variation': 0, 'aspect_ratio': 0,
            'gei_mean': 0, 'gei_std': 0, 'gei_energy': 0, 'gei_centroid_x': 0, 'gei_centroid_y': 0
        }
    
    def create_feature_dataset(self) -> tuple:
        """Create feature dataset"""
        print("🔍 Extracting essential gait features...")
        
        all_features = []
        labels = []
        
        for subject_id, subject_data in self.sequences.items():
            subject_label = int(subject_id)
            
            for condition, condition_data in subject_data.items():
                for angle, sequence in condition_data.items():
                    # Extract features
                    features = self.extract_quick_features(sequence)
                    
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

class QuickClassifier:
    """Quick classifier optimized for speed"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        
    def prepare_data(self, features_df, labels, test_size=0.2):
        """Prepare data quickly"""
        print("🔧 Preparing data for training...")
        
        # Remove metadata columns
        feature_columns = [col for col in features_df.columns 
                          if col not in ['subject_id', 'condition', 'angle']]
        X = features_df[feature_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            X = X.drop(columns=constant_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=self.random_state, 
            stratify=labels, shuffle=True
        )
        
        # Feature selection
        k = min(15, X_train.shape[1])
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
        """Initialize quick models"""
        print("🚀 Initializing models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', C=10.0, gamma='scale',
                random_state=self.random_state, probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=3, weights='distance'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', alpha=0.001, max_iter=500,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state,
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
    
    def cross_validate_models(self, X_train, y_train, cv=3):
        """Quick cross-validation"""
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

def run_quick_training(dataset_path: str):
    """Run quick CASIA-B training"""
    print("🎯 QUICK CASIA-B GAIT RECOGNITION TRAINING")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print("Goal: Achieve good accuracy for person identification (5-10 minutes)")
    print()
    
    # Initialize processor
    processor = QuickCASIABLoader(dataset_path)
    
    # Load sequences quickly
    total_sequences = processor.load_gait_sequences(max_subjects=25, max_sequences_per_subject=2)
    if total_sequences == 0:
        print("❌ No sequences loaded. Please check the dataset.")
        return None
    
    # Extract features
    features_df, labels = processor.create_feature_dataset()
    
    # Initialize classifier
    classifier = QuickClassifier()
    
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
    print("🎯 TRAINING SUMMARY")
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
        print("🎉 EXCELLENT! 90%+ accuracy achieved!")
        print("✅ The model can reliably identify who is walking!")
    elif best_test_score >= 0.80:
        print("✅ GOOD! 80%+ accuracy achieved!")
        print("✅ The model can effectively identify who is walking!")
    elif best_test_score >= 0.70:
        print("✅ DECENT! 70%+ accuracy achieved!")
        print("✅ The model shows good person identification capability!")
    else:
        print("⚠️  Performance below 70%")
        print("💡 Consider: more data, better features, or model tuning")
    
    print("\n🎯 What the model predicts:")
    print("   • Input: Gait sequence (video frames of walking)")
    print("   • Output: Person identity (Subject_001, Subject_087, etc.)")
    print("   • Accuracy: How often the model correctly identifies the person")
    
    print("\n💡 Next Steps:")
    print("   1. Test the model on new gait sequences")
    print("   2. Fine-tune hyperparameters for better performance")
    print("   3. Add more training data if available")
    print("   4. Deploy for real-world gait recognition")
    
    return classifier, final_results, cv_results

if __name__ == "__main__":
    # Run quick training
    dataset_path = "./casia-b"
    classifier, results, cv_results = run_quick_training(dataset_path)

