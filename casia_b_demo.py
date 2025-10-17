"""
Complete CASIA-B Dataset Demo for Gait-Based Biometric Recognition
Demonstrates the full pipeline with CASIA-B style data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from casia_b_loader import create_casia_b_demo

def run_casia_b_experiment():
    """
    Run complete experiment with CASIA-B style data
    """
    print("=" * 70)
    print("   CASIA-B GAIT-BASED BIOMETRIC RECOGNITION EXPERIMENT")
    print("=" * 70)
    print()
    
    # Step 1: Load CASIA-B style data
    print("Step 1: Loading CASIA-B style dataset...")
    dataset, features_df, labels = create_casia_b_demo()
    
    print(f"✓ Loaded {len(dataset)} subjects")
    print(f"✓ Generated {len(labels)} gait sequences")
    print(f"✓ Extracted {features_df.shape[1]} features")
    
    # Step 2: Data preparation
    print("\nStep 2: Preparing data for machine learning...")
    
    # Remove non-feature columns
    feature_columns = [col for col in features_df.columns if col not in ['subject_id', 'condition', 'angle']]
    X = features_df[feature_columns]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Training set: {X_train_scaled.shape}")
    print(f"✓ Test set: {X_test_scaled.shape}")
    print(f"✓ Features used: {len(feature_columns)}")
    
    # Step 3: Train models
    print("\nStep 3: Training machine learning models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"    ✓ {name}: {accuracy:.4f}")
        except Exception as e:
            print(f"    ✗ {name}: Error - {str(e)}")
    
    # Step 4: Cross-validation
    print("\nStep 4: Cross-validation results...")
    
    cv_results = {}
    for name, model in models.items():
        if name in results:
            try:
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                cv_results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                }
                print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"  {name}: CV Error - {str(e)}")
    
    # Step 5: Results analysis
    print("\nStep 5: Results Analysis")
    print("=" * 50)
    
    if cv_results:
        # Sort by performance
        sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        
        print("Model Performance (Cross-Validation):")
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"  {i}. {name:<20}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        
        best_model = sorted_results[0][0]
        best_score = sorted_results[0][1]['mean_score']
        
        print(f"\n🏆 Best Model: {best_model} with {best_score:.4f} accuracy")
        
        # Performance interpretation
        print("\nPerformance Analysis:")
        if best_score > 0.9:
            print("  ✓ Excellent performance! CASIA-B data shows strong biometric recognition capability.")
        elif best_score > 0.8:
            print("  ✓ Good performance. The system can effectively distinguish between subjects using CASIA-B data.")
        elif best_score > 0.7:
            print("  ⚠ Moderate performance. Consider feature engineering or larger dataset.")
        elif best_score > 0.5:
            print("  ⚠ Lower performance. May need more diverse features or better preprocessing.")
        else:
            print("  ⚠ Poor performance. Consider data quality or model selection.")
    
    # Step 6: Feature importance
    print("\nStep 6: Feature Importance Analysis")
    print("-" * 40)
    
    if 'Random Forest' in models and 'Random Forest' in results:
        rf_model = models['Random Forest']
        rf_model.fit(X_train_scaled, y_train)
        
        feature_importance = rf_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:]
        
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(reversed(top_features_idx)):
            feature_name = feature_columns[idx]
            importance = feature_importance[idx]
            print(f"  {i+1:2d}. {feature_name:<35}: {importance:.4f}")
    
    # Step 7: Visualization
    print("\nStep 7: Creating visualizations...")
    
    try:
        # Model performance comparison
        plt.figure(figsize=(12, 6))
        
        if cv_results:
            model_names = list(cv_results.keys())
            accuracies = [cv_results[name]['mean_score'] for name in model_names]
            
            bars = plt.bar(model_names, accuracies, color='lightcoral', alpha=0.7)
            plt.title('CASIA-B Gait Recognition Model Performance', fontsize=14, fontweight='bold')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            print("✓ Performance comparison chart displayed")
        
        # Feature importance plot
        if 'Random Forest' in models and 'Random Forest' in results:
            plt.figure(figsize=(12, 8))
            
            # Get top 15 features
            top_15_idx = np.argsort(feature_importance)[-15:]
            top_features = [feature_columns[i] for i in top_15_idx]
            top_importances = feature_importance[top_15_idx]
            
            bars = plt.barh(range(len(top_features)), top_importances, color='skyblue', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features for Gait Recognition', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, top_importances)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{imp:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            print("✓ Feature importance chart displayed")
    
    except Exception as e:
        print(f"⚠ Visualization error: {str(e)}")
    
    # Step 8: CASIA-B specific analysis
    print("\nStep 8: CASIA-B Dataset Analysis")
    print("-" * 40)
    
    # Analyze by condition
    condition_analysis = features_df.groupby('condition').size()
    print("Sequences by condition:")
    for condition, count in condition_analysis.items():
        print(f"  {condition}: {count} sequences")
    
    # Analyze by angle
    angle_analysis = features_df.groupby('angle').size()
    print(f"\nSequences by viewing angle:")
    for angle, count in angle_analysis.items():
        print(f"  {angle}°: {count} sequences")
    
    print("\n" + "=" * 70)
    print("   CASIA-B EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n💡 CASIA-B Integration Insights:")
    print("  • CASIA-B dataset provides excellent real-world gait data")
    print("  • Multiple viewing angles improve recognition robustness")
    print("  • Different conditions (normal/bag/coat) test generalization")
    print("  • Silhouette-based features are effective for gait recognition")
    
    print("\n🚀 Next Steps for Real CASIA-B Dataset:")
    print("  1. Download CASIA-B dataset from official source")
    print("  2. Implement silhouette preprocessing pipeline")
    print("  3. Extract real gait features from binary images")
    print("  4. Train on full 124-subject dataset")
    print("  5. Evaluate across all viewing angles and conditions")
    print("  6. Compare with state-of-the-art methods")
    
    return {
        'dataset': dataset,
        'features': features_df,
        'labels': labels,
        'results': results,
        'cv_results': cv_results,
        'feature_columns': feature_columns
    }

if __name__ == "__main__":
    result = run_casia_b_experiment()
