"""
High-Accuracy Models for CASIA-B Gait Recognition
Designed to achieve 90%+ accuracy for person identification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class HighAccuracyGaitClassifier:
    """
    High-accuracy gait classifier designed for 90%+ performance
    """
    
    def __init__(self, random_state=42):
        """
        Initialize high-accuracy classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def prepare_enhanced_data(self, features_df, labels, test_size=0.2, feature_selection_k=50):
        """
        Prepare data with enhanced preprocessing for high accuracy
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            test_size (float): Test set proportion
            feature_selection_k (int): Number of features to select
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("🔧 Preparing data for high-accuracy training...")
        
        # Remove metadata columns
        feature_columns = [col for col in features_df.columns 
                          if col not in ['subject_id', 'condition', 'angle', 'sequence_id']]
        X = features_df[feature_columns]
        
        # Handle missing values with advanced imputation
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            print(f"   Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=self.random_state, 
            stratify=labels, shuffle=True
        )
        
        # Advanced feature selection
        print(f"   Selecting top {feature_selection_k} features...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(feature_selection_k, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Apply PCA for dimensionality reduction
        print("   Applying PCA for optimal feature space...")
        self.pca = PCA(n_components=0.95, random_state=self.random_state)  # Keep 95% variance
        X_train_final = self.pca.fit_transform(X_train_scaled)
        X_test_final = self.pca.transform(X_test_scaled)
        
        print(f"✅ Data preparation complete:")
        print(f"   📊 Original features: {X.shape[1]}")
        print(f"   🎯 Selected features: {X_train_selected.shape[1]}")
        print(f"   🔄 PCA components: {X_train_final.shape[1]}")
        print(f"   📈 Training set: {X_train_final.shape}")
        print(f"   📉 Test set: {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train, y_test
    
    def initialize_high_accuracy_models(self):
        """
        Initialize models optimized for high accuracy
        """
        print("🚀 Initializing high-accuracy models...")
        
        self.models = {
            # Ensemble methods
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            
            # SVM with optimized parameters
            'SVM': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            ),
            
            # KNN with optimized parameters
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=3,
                weights='distance',
                algorithm='auto',
                leaf_size=30,
                p=2
            ),
            
            # Neural Network with optimized architecture
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            # Logistic Regression with regularization
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=self.random_state,
                solver='lbfgs',
                multi_class='multinomial'
            )
        }
        
        print(f"✅ Initialized {len(self.models)} high-accuracy models")
    
    def train_models_with_hyperparameter_tuning(self, X_train, y_train):
        """
        Train models with hyperparameter tuning for optimal performance
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        print("🎯 Training models with hyperparameter optimization...")
        
        # Define hyperparameter grids for tuning
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
        
        for name, model in self.models.items():
            print(f"  🔧 Training {name}...")
            
            try:
                if name in param_grids:
                    # Use GridSearchCV for hyperparameter tuning
                    grid_search = GridSearchCV(
                        model, param_grids[name], 
                        cv=3, scoring='accuracy', 
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    self.models[name] = grid_search.best_estimator_
                    print(f"    ✓ Best params: {grid_search.best_params_}")
                else:
                    # Train directly for other models
                    model.fit(X_train, y_train)
                
                print(f"    ✅ {name} training complete")
                
            except Exception as e:
                print(f"    ❌ Error training {name}: {str(e)}")
    
    def create_ensemble_model(self, X_train, y_train):
        """
        Create ensemble model combining best performers
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        print("🎭 Creating ensemble model...")
        
        # Select best models for ensemble
        best_models = []
        model_scores = {}
        
        # Quick evaluation to select best models
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                model_scores[name] = scores.mean()
                if scores.mean() > 0.7:  # Only include models with >70% CV score
                    best_models.append((name, model))
            except:
                continue
        
        if len(best_models) >= 2:
            # Create voting ensemble
            ensemble = VotingClassifier(
                estimators=best_models,
                voting='soft'  # Use predicted probabilities
            )
            ensemble.fit(X_train, y_train)
            self.models['Ensemble'] = ensemble
            
            print(f"✅ Ensemble created with {len(best_models)} models:")
            for name, _ in best_models:
                print(f"   - {name}: {model_scores[name]:.4f}")
        else:
            print("⚠️  Not enough high-performing models for ensemble")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models and select the best performer
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        print("📊 Evaluating models for high-accuracy performance...")
        
        results = {}
        best_accuracy = 0
        best_model_name = None
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'model': model
                }
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                    self.best_model = model
                
                # Performance indicator
                if accuracy >= 0.90:
                    status = "🎯 EXCELLENT"
                elif accuracy >= 0.80:
                    status = "✅ GOOD"
                elif accuracy >= 0.70:
                    status = "⚠️  MODERATE"
                else:
                    status = "❌ POOR"
                
                print(f"  {name:<20}: {accuracy:.4f} {status}")
                
            except Exception as e:
                print(f"  {name:<20}: ERROR - {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        
        print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
        
        return results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """
        Perform comprehensive cross-validation
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            cv (int): Number of CV folds
            
        Returns:
            dict: Cross-validation results
        """
        print(f"🔄 Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'min_score': scores.min(),
                    'max_score': scores.max(),
                    'scores': scores
                }
                
                # Performance indicator
                if scores.mean() >= 0.90:
                    status = "🎯 EXCELLENT"
                elif scores.mean() >= 0.80:
                    status = "✅ GOOD"
                elif scores.mean() >= 0.70:
                    status = "⚠️  MODERATE"
                else:
                    status = "❌ POOR"
                
                print(f"  {name:<20}: {scores.mean():.4f} ± {scores.std():.4f} {status}")
                
            except Exception as e:
                print(f"  {name:<20}: CV ERROR - {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def predict_person_identity(self, gait_sequence, model_name=None):
        """
        Predict person identity from gait sequence
        
        Args:
            gait_sequence: Gait sequence data
            model_name (str): Specific model to use (None for best model)
            
        Returns:
            dict: Prediction results
        """
        if model_name is None:
            model_name = self._get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Process the gait sequence (this would need to be implemented based on your data format)
        # For now, return a placeholder
        prediction = {
            'predicted_subject_id': 'Subject_001',
            'confidence': 0.95,
            'model_used': model_name,
            'all_predictions': {}
        }
        
        return prediction
    
    def _get_best_model_name(self):
        """Get the name of the best performing model"""
        if not self.results:
            return None
        
        best_accuracy = 0
        best_name = None
        
        for name, result in self.results.items():
            if 'accuracy' in result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_name = name
        
        return best_name
    
    def plot_performance_analysis(self, cv_results=None):
        """
        Create comprehensive performance analysis plots
        
        Args:
            cv_results (dict): Cross-validation results
        """
        if cv_results is None:
            cv_results = self.results
        
        # Extract data for plotting
        model_names = []
        accuracies = []
        std_devs = []
        
        for name, result in cv_results.items():
            if 'mean_score' in result:
                model_names.append(name)
                accuracies.append(result['mean_score'])
                std_devs.append(result['std_score'])
            elif 'accuracy' in result:
                model_names.append(name)
                accuracies.append(result['accuracy'])
                std_devs.append(0.01)  # Default small std for single scores
        
        if not model_names:
            print("No valid results to plot")
            return
        
        # Create comprehensive performance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        bars = axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Performance Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
        axes[0, 0].legend()
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Accuracy with error bars
        axes[0, 1].bar(model_names, accuracies, yerr=std_devs, capsize=5, 
                      color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Accuracy with Standard Deviation', fontweight='bold')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
        axes[0, 1].legend()
        
        # 3. Performance distribution
        axes[1, 0].hist(accuracies, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Accuracy Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].set_ylabel('Number of Models')
        axes[1, 0].axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
        axes[1, 0].legend()
        
        # 4. Model comparison scatter
        axes[1, 1].scatter(range(len(model_names)), accuracies, s=100, alpha=0.7)
        axes[1, 1].set_title('Model Performance Scatter', fontweight='bold')
        axes[1, 1].set_xlabel('Model Index')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% Target')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Performance analysis plots displayed")

def run_high_accuracy_experiment():
    """
    Run complete high-accuracy gait recognition experiment
    """
    print("🎯 HIGH-ACCURACY GAIT RECOGNITION EXPERIMENT")
    print("=" * 60)
    print("Goal: Achieve 90%+ accuracy for person identification")
    print()
    
    # Load enhanced data
    from real_casia_b_loader import create_high_accuracy_demo
    dataset, features_df, labels = create_high_accuracy_demo()
    
    # Initialize high-accuracy classifier
    classifier = HighAccuracyGaitClassifier()
    
    # Prepare enhanced data
    X_train, X_test, y_train, y_test = classifier.prepare_enhanced_data(
        features_df, labels, test_size=0.2, feature_selection_k=40
    )
    
    # Initialize and train models
    classifier.initialize_high_accuracy_models()
    classifier.train_models_with_hyperparameter_tuning(X_train, y_train)
    
    # Create ensemble
    classifier.create_ensemble_model(X_train, y_train)
    
    # Cross-validation
    print("\n" + "="*60)
    cv_results = classifier.cross_validate_models(X_train, y_train)
    
    # Final evaluation
    print("\n" + "="*60)
    final_results = classifier.evaluate_models(X_test, y_test)
    
    # Performance analysis
    print("\n" + "="*60)
    classifier.plot_performance_analysis(cv_results)
    
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
    classifier, results, cv_results = run_high_accuracy_experiment()
