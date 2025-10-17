"""
Main Script for Gait-Based Biometric Recognition System
Complete pipeline for gait recognition using machine learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import GaitDataPreprocessor, create_sample_dataset
from feature_extraction import GaitFeatureExtractor, create_feature_dataset
from ml_models import GaitBiometricClassifier, run_complete_experiment
from visualization import GaitVisualizer

def main():
    """
    Main function to run the complete gait biometric recognition pipeline
    """
    print("=" * 60)
    print("   GAIT-BASED BIOMETRIC RECOGNITION SYSTEM")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        'n_subjects': 30,
        'n_samples_per_subject': 50,
        'sampling_rate': 50,
        'test_size': 0.2,
        'feature_selection_k': 50,
        'random_state': 42
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Step 1: Data Generation and Preprocessing
        print("STEP 1: Data Generation and Preprocessing")
        print("-" * 40)
        
        preprocessor = GaitDataPreprocessor(sampling_rate=config['sampling_rate'])
        
        # Generate synthetic gait data
        print("Generating synthetic gait data...")
        raw_data = preprocessor.load_synthetic_data(
            n_subjects=config['n_subjects'], 
            n_samples_per_subject=config['n_samples_per_subject']
        )
        print(f"✓ Generated data for {len(raw_data)} subjects")
        
        # Segment gait cycles
        print("Segmenting gait cycles...")
        segmented_data = preprocessor.segment_gait_cycles(raw_data)
        total_cycles = sum(len(cycles) for cycles in segmented_data.values())
        print(f"✓ Segmented {total_cycles} gait cycles")
        
        # Normalize cycles
        print("Normalizing gait cycles...")
        normalized_data = preprocessor.normalize_cycles(segmented_data)
        print(f"✓ Normalized all gait cycles to standard length")
        
        print()
        
        # Step 2: Feature Extraction
        print("STEP 2: Feature Extraction")
        print("-" * 40)
        
        extractor = GaitFeatureExtractor(sampling_rate=config['sampling_rate'])
        
        print("Extracting features from gait cycles...")
        features_df, labels = extractor.extract_features_from_dataset(normalized_data)
        print(f"✓ Extracted {features_df.shape[1]} features from {len(labels)} cycles")
        print(f"✓ Features shape: {features_df.shape}")
        print(f"✓ Number of unique subjects: {len(np.unique(labels))}")
        
        # Display feature summary
        print("\nFeature Categories:")
        feature_categories = {
            'Temporal': [col for col in features_df.columns if any(x in col for x in ['mean', 'std', 'var', 'skew', 'kurtosis', 'rms', 'energy', 'entropy'])],
            'Frequency': [col for col in features_df.columns if any(x in col for x in ['freq', 'spectral', 'power'])],
            'Waveform': [col for col in features_df.columns if any(x in col for x in ['symmetry', 'regularity', 'smoothness'])],
            'Phase': [col for col in features_df.columns if any(x in col for x in ['phase', 'stance', 'swing', 'support'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
        
        print()
        
        # Step 3: Machine Learning Models
        print("STEP 3: Machine Learning Models")
        print("-" * 40)
        
        # Initialize classifier
        classifier = GaitBiometricClassifier(random_state=config['random_state'])
        
        # Prepare data
        print("Preparing data for machine learning...")
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            features_df, labels, test_size=config['test_size']
        )
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Feature selection
        print(f"Performing feature selection (selecting top {config['feature_selection_k']} features)...")
        X_train_selected = classifier.feature_selection(X_train, y_train, k=config['feature_selection_k'])
        X_test_selected = classifier.feature_selector.transform(X_test)
        print(f"✓ Selected {X_train_selected.shape[1]} features")
        
        # Train models
        print("Training machine learning models...")
        classifier.train_models(X_train_selected, y_train)
        
        # Evaluate models
        print("Evaluating models on test set...")
        results = classifier.evaluate_models(X_test_selected, y_test)
        
        # Cross-validation
        print("Performing cross-validation...")
        cv_results = classifier.cross_validate_models(X_train_selected, y_train)
        
        # Train deep learning model
        print("Training deep learning model...")
        try:
            dl_model, history = classifier.train_deep_learning_model(
                X_train_selected, y_train, X_test_selected, y_test, epochs=50
            )
            dl_loss, dl_accuracy = dl_model.evaluate(X_test_selected, y_test, verbose=0)
            results['Deep Learning'] = {'accuracy': dl_accuracy, 'model': dl_model}
            print(f"✓ Deep Learning: Accuracy = {dl_accuracy:.4f}")
        except Exception as e:
            print(f"⚠ Deep Learning training failed: {str(e)}")
        
        print()
        
        # Step 4: Results Analysis
        print("STEP 4: Results Analysis")
        print("-" * 40)
        
        print("Model Performance Summary:")
        print("=" * 50)
        
        # Sort results by accuracy
        sorted_results = sorted(cv_results.items(), key=lambda x: x[1].get('mean_score', 0), reverse=True)
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            if 'mean_score' in result:
                accuracy = result['mean_score']
                std = result['std_score']
                print(f"{i:2d}. {model_name:<20}: {accuracy:.4f} ± {std:.4f}")
            elif 'accuracy' in result:
                accuracy = result['accuracy']
                print(f"{i:2d}. {model_name:<20}: {accuracy:.4f}")
        
        # Best model
        best_model_name = sorted_results[0][0]
        best_accuracy = sorted_results[0][1].get('mean_score', sorted_results[0][1].get('accuracy', 0))
        
        print()
        print(f"🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
        
        print()
        
        # Step 5: Visualization
        print("STEP 5: Visualization")
        print("-" * 40)
        
        visualizer = GaitVisualizer()
        
        # Create sample visualizations
        print("Creating visualizations...")
        
        # Sample gait cycle plot
        sample_subject = list(normalized_data.keys())[0]
        if normalized_data[sample_subject]:
            print("  - Sample gait cycle visualization")
            visualizer.plot_gait_cycle(normalized_data[sample_subject][0], sample_subject)
        
        # Feature correlation
        print("  - Feature correlation matrix")
        visualizer.plot_feature_correlation_matrix(features_df)
        
        # PCA visualization
        print("  - PCA visualization")
        visualizer.plot_pca_visualization(features_df, labels)
        
        # Model performance comparison
        print("  - Model performance comparison")
        visualizer.plot_results(cv_results)
        
        print()
        
        # Step 6: Summary and Recommendations
        print("STEP 6: Summary and Recommendations")
        print("-" * 40)
        
        print("System Performance Summary:")
        print(f"  • Dataset: {config['n_subjects']} subjects, {total_cycles} gait cycles")
        print(f"  • Features: {features_df.shape[1]} extracted features")
        print(f"  • Best Model: {best_model_name}")
        print(f"  • Accuracy: {best_accuracy:.4f}")
        
        print("\nRecommendations:")
        if best_accuracy > 0.9:
            print("  ✓ Excellent performance! The system is ready for deployment.")
        elif best_accuracy > 0.8:
            print("  ✓ Good performance. Consider fine-tuning hyperparameters.")
        elif best_accuracy > 0.7:
            print("  ⚠ Moderate performance. Consider:")
            print("    - Increasing dataset size")
            print("    - Feature engineering")
            print("    - Model ensemble")
        else:
            print("  ⚠ Low performance. Consider:")
            print("    - Data quality improvement")
            print("    - Feature selection refinement")
            print("    - Different preprocessing techniques")
        
        print("\nNext Steps:")
        print("  1. Collect real gait data from sensors")
        print("  2. Validate on larger dataset")
        print("  3. Implement real-time processing")
        print("  4. Deploy for biometric authentication")
        
        print()
        print("=" * 60)
        print("   EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return {
            'dataset': normalized_data,
            'features': features_df,
            'labels': labels,
            'classifier': classifier,
            'results': results,
            'cv_results': cv_results,
            'config': config
        }
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_quick_demo():
    """
    Run a quick demonstration with smaller dataset
    """
    print("🚀 Running Quick Demo...")
    
    # Override config for quick demo
    import sys
    sys.argv = ['main.py', '--quick']
    
    return main()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick demo with smaller dataset
        print("Running quick demonstration...")
        result = run_quick_demo()
    else:
        # Full experiment
        result = main()
    
    if result:
        print("\n🎉 Experiment completed! Check the results above.")
        print("💡 You can now explore the individual modules:")
        print("   - data_preprocessing.py: For data handling")
        print("   - feature_extraction.py: For feature engineering")
        print("   - ml_models.py: For machine learning models")
        print("   - visualization.py: For data visualization")
    else:
        print("\n❌ Experiment failed. Please check the error messages above.")
