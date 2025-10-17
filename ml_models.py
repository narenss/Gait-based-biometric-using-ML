"""
Machine Learning Models for Gait-Based Biometric Recognition
Implements various ML algorithms for gait recognition
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GaitBiometricClassifier:
    """
    Comprehensive gait biometric classification system
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the classifier
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.models = {}
        self.results = {}
        
    def prepare_data(self, features_df, labels, test_size=0.2):
        """
        Prepare data for training and testing
        
        Args:
            features_df (pd.DataFrame): Feature matrix
            labels (np.array): Subject labels
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def feature_selection(self, X_train, y_train, k=50):
        """
        Select best features using statistical tests
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            k (int): Number of features to select
            
        Returns:
            np.array: Selected features
        """
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        return X_train_selected
    
    def initialize_models(self):
        """
        Initialize various machine learning models
        """
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, 
                weights='distance'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=self.random_state
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=self.random_state,
                verbose=-1
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                activation='relu', 
                solver='adam', 
                alpha=0.001, 
                batch_size='auto', 
                learning_rate='constant', 
                learning_rate_init=0.001, 
                max_iter=500, 
                random_state=self.random_state
            ),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            )
        }
    
    def train_models(self, X_train, y_train):
        """
        Train all models
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'model': model
                }
                
                print(f"{name}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """
        Perform cross-validation on all models
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
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
                print(f"{name} CV: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            except Exception as e:
                print(f"Error in CV for {name}: {str(e)}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def create_deep_learning_model(self, input_dim, num_classes):
        """
        Create a deep learning model using TensorFlow/Keras
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of classes (subjects)
            
        Returns:
            keras.Model: Compiled deep learning model
        """
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_deep_learning_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train deep learning model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            X_test (np.array): Test features
            y_test (np.array): Test labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            keras.Model: Trained model
        """
        num_classes = len(np.unique(y_train))
        model = self.create_deep_learning_model(X_train.shape[1], num_classes)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.models['Deep Learning'] = model
        return model, history
    
    def plot_results(self, cv_results=None):
        """
        Plot model comparison results
        
        Args:
            cv_results (dict): Cross-validation results
        """
        if cv_results is None:
            cv_results = self.results
        
        # Extract accuracies
        model_names = []
        accuracies = []
        
        for name, result in cv_results.items():
            if 'mean_score' in result:
                model_names.append(name)
                accuracies.append(result['mean_score'])
            elif 'accuracy' in result:
                model_names.append(name)
                accuracies.append(result['accuracy'])
        
        if not model_names:
            print("No valid results to plot")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred):
        """
        Plot confusion matrix for a specific model
        
        Args:
            model_name (str): Name of the model
            y_test (np.array): True labels
            y_pred (np.array): Predicted labels
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Subject', fontsize=12)
        plt.ylabel('True Subject', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, model_name='Random Forest'):
        """
        Get feature importance from tree-based models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            np.array: Feature importance scores
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None

def run_complete_experiment():
    """
    Run a complete gait biometric recognition experiment
    """
    from feature_extraction import create_feature_dataset
    
    print("=== Gait-Based Biometric Recognition Experiment ===\n")
    
    # Load data
    print("1. Loading and preprocessing data...")
    features_df, labels = create_feature_dataset()
    print(f"   Features shape: {features_df.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Number of subjects: {len(np.unique(labels))}")
    
    # Initialize classifier
    print("\n2. Initializing classifier...")
    classifier = GaitBiometricClassifier()
    
    # Prepare data
    print("\n3. Preparing data...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(features_df, labels)
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Feature selection
    print("\n4. Performing feature selection...")
    X_train_selected = classifier.feature_selection(X_train, y_train, k=50)
    X_test_selected = classifier.feature_selector.transform(X_test)
    print(f"   Selected features: {X_train_selected.shape[1]}")
    
    # Train traditional ML models
    print("\n5. Training machine learning models...")
    classifier.train_models(X_train_selected, y_train)
    
    # Evaluate models
    print("\n6. Evaluating models...")
    results = classifier.evaluate_models(X_test_selected, y_test)
    
    # Cross-validation
    print("\n7. Performing cross-validation...")
    cv_results = classifier.cross_validate_models(X_train_selected, y_train)
    
    # Train deep learning model
    print("\n8. Training deep learning model...")
    try:
        dl_model, history = classifier.train_deep_learning_model(
            X_train_selected, y_train, X_test_selected, y_test, epochs=50
        )
        
        # Evaluate deep learning model
        dl_loss, dl_accuracy = dl_model.evaluate(X_test_selected, y_test, verbose=0)
        results['Deep Learning'] = {
            'accuracy': dl_accuracy,
            'loss': dl_loss,
            'model': dl_model
        }
        print(f"Deep Learning: Accuracy = {dl_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error training deep learning model: {str(e)}")
    
    # Plot results
    print("\n9. Plotting results...")
    classifier.plot_results(cv_results)
    
    # Summary
    print("\n=== EXPERIMENT SUMMARY ===")
    print("Model Performance (Cross-Validation):")
    for name, result in cv_results.items():
        if 'mean_score' in result:
            print(f"  {name}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
    
    return classifier, results, cv_results

if __name__ == "__main__":
    # Run complete experiment
    classifier, results, cv_results = run_complete_experiment()
