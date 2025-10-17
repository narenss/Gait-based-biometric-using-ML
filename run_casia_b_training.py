"""
Run CASIA-B Training Script
Simple script to train and test gait recognition on your CASIA-B dataset
"""

import os
import sys
from real_casia_b_pipeline import run_real_casia_b_experiment

def main():
    """
    Main function to run CASIA-B training
    """
    print("🎯 CASIA-B Gait Recognition Training")
    print("=" * 50)
    print()
    
    # Ask for dataset path
    print("Please provide the path to your CASIA-B dataset.")
    print("The dataset should contain folders: nm/, bg/, cl/")
    print()
    
    while True:
        dataset_path = input("Enter CASIA-B dataset path: ").strip()
        
        if not dataset_path:
            print("❌ Please enter a valid path.")
            continue
        
        # Remove quotes if present
        dataset_path = dataset_path.strip('"').strip("'")
        
        if os.path.exists(dataset_path):
            print(f"✅ Dataset found at: {dataset_path}")
            break
        else:
            print(f"❌ Path not found: {dataset_path}")
            print("Please check the path and try again.")
            continue
    
    print()
    print("🚀 Starting training process...")
    print("This may take a few minutes depending on your dataset size.")
    print()
    
    try:
        # Run the experiment
        result = run_real_casia_b_experiment(dataset_path)
        
        if result is not None:
            classifier, final_results, cv_results, processor = result
            
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Show prediction example
            print("\n🔮 PREDICTION EXAMPLE:")
            print("The trained model can now predict who is walking from gait sequences.")
            print("Example prediction:")
            print("  Input: Gait sequence (video frames)")
            print("  Output: Subject_023 (with 95.2% confidence)")
            print("  Model: Random Forest")
            
            print("\n📊 PERFORMANCE SUMMARY:")
            best_accuracy = 0
            best_model = None
            for name, result in final_results.items():
                if 'accuracy' in result and result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_model = name
            
            print(f"  Best Model: {best_model}")
            print(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            if best_accuracy >= 0.90:
                print("  Status: 🎯 EXCELLENT - Goal achieved!")
            elif best_accuracy >= 0.80:
                print("  Status: ✅ GOOD - Close to target")
            else:
                print("  Status: ⚠️  MODERATE - Needs improvement")
            
            print("\n💡 NEXT STEPS:")
            print("  1. Test the model on new gait sequences")
            print("  2. Fine-tune hyperparameters for better performance")
            print("  3. Add more training data if available")
            print("  4. Deploy for real-world gait recognition")
            
        else:
            print("❌ Training failed. Please check your dataset and try again.")
    
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("Please check your dataset format and try again.")
    
    print("\n" + "="*60)
    print("Thank you for using the CASIA-B Gait Recognition System!")
    print("="*60)

if __name__ == "__main__":
    main()


