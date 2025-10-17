"""
Gait Data Preprocessing Module
Handles data loading, cleaning, and preprocessing for gait-based biometric recognition
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class GaitDataPreprocessor:
    """
    A comprehensive preprocessing pipeline for gait data
    """
    
    def __init__(self, sampling_rate=50):
        """
        Initialize the preprocessor
        
        Args:
            sampling_rate (int): Sampling rate of the gait data in Hz
        """
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        
    def load_synthetic_data(self, n_subjects=50, n_samples_per_subject=100):
        """
        Generate synthetic gait data for demonstration purposes
        In a real scenario, this would load actual sensor data
        
        Args:
            n_subjects (int): Number of subjects
            n_samples_per_subject (int): Number of gait cycles per subject
            
        Returns:
            dict: Dictionary containing gait data for each subject
        """
        np.random.seed(42)
        gait_data = {}
        
        for subject_id in range(n_subjects):
            subject_data = []
            
            for cycle in range(n_samples_per_subject):
                # Generate synthetic gait cycle data
                # Real gait data would come from accelerometers, gyroscopes, or pressure sensors
                
                # Simulate a gait cycle (approximately 1.2 seconds)
                cycle_duration = 1.2 + np.random.normal(0, 0.1)
                time_points = np.linspace(0, cycle_duration, int(cycle_duration * self.sampling_rate))
                
                # Generate synthetic sensor data with subject-specific characteristics
                subject_signature = np.random.normal(0, 0.5)  # Unique to each subject
                
                # Vertical acceleration (most important for gait)
                vertical_acc = (np.sin(2 * np.pi * time_points / cycle_duration * 2) + 
                              subject_signature * np.sin(2 * np.pi * time_points / cycle_duration * 4) +
                              np.random.normal(0, 0.1, len(time_points)))
                
                # Anterior-posterior acceleration
                ap_acc = (0.5 * np.cos(2 * np.pi * time_points / cycle_duration * 2) + 
                         subject_signature * 0.3 * np.cos(2 * np.pi * time_points / cycle_duration * 3) +
                         np.random.normal(0, 0.05, len(time_points)))
                
                # Medial-lateral acceleration
                ml_acc = (0.3 * np.sin(2 * np.pi * time_points / cycle_duration * 1.5) + 
                         subject_signature * 0.2 * np.sin(2 * np.pi * time_points / cycle_duration * 2.5) +
                         np.random.normal(0, 0.03, len(time_points)))
                
                cycle_data = {
                    'time': time_points,
                    'vertical_acc': vertical_acc,
                    'ap_acc': ap_acc,
                    'ml_acc': ml_acc,
                    'subject_id': subject_id,
                    'cycle_id': cycle
                }
                subject_data.append(cycle_data)
            
            gait_data[f'subject_{subject_id}'] = subject_data
            
        return gait_data
    
    def filter_data(self, data, filter_type='butter', cutoff_freq=10):
        """
        Apply low-pass filter to remove high-frequency noise
        
        Args:
            data (numpy.ndarray): Input signal data
            filter_type (str): Type of filter ('butter', 'cheby1', 'ellip')
            cutoff_freq (float): Cutoff frequency in Hz
            
        Returns:
            numpy.ndarray: Filtered data
        """
        nyquist = self.sampling_rate / 2
        normal_cutoff = cutoff_freq / nyquist
        
        if filter_type == 'butter':
            b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        elif filter_type == 'cheby1':
            b, a = signal.cheby1(4, 0.5, normal_cutoff, btype='low', analog=False)
        elif filter_type == 'ellip':
            b, a = signal.ellip(4, 0.5, 40, normal_cutoff, btype='low', analog=False)
        else:
            raise ValueError("Filter type must be 'butter', 'cheby1', or 'ellip'")
        
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    def detect_heel_strikes(self, vertical_acc, threshold=0.7):
        """
        Detect heel strikes using peak detection on vertical acceleration
        
        Args:
            vertical_acc (numpy.ndarray): Vertical acceleration signal
            threshold (float): Threshold for peak detection
            
        Returns:
            numpy.ndarray: Indices of detected heel strikes
        """
        # Find peaks in vertical acceleration
        peaks, _ = signal.find_peaks(vertical_acc, height=np.max(vertical_acc) * threshold)
        return peaks
    
    def segment_gait_cycles(self, data_dict):
        """
        Segment continuous gait data into individual gait cycles
        
        Args:
            data_dict (dict): Dictionary containing gait data for each subject
            
        Returns:
            dict: Dictionary with segmented gait cycles
        """
        segmented_data = {}
        
        for subject_id, subject_data in data_dict.items():
            segmented_cycles = []
            
            for cycle_data in subject_data:
                # Apply filtering
                filtered_vertical = self.filter_data(cycle_data['vertical_acc'])
                filtered_ap = self.filter_data(cycle_data['ap_acc'])
                filtered_ml = self.filter_data(cycle_data['ml_acc'])
                
                # Detect heel strikes
                heel_strikes = self.detect_heel_strikes(filtered_vertical)
                
                if len(heel_strikes) >= 2:
                    # Segment based on heel strikes
                    for i in range(len(heel_strikes) - 1):
                        start_idx = heel_strikes[i]
                        end_idx = heel_strikes[i + 1]
                        
                        cycle_segment = {
                            'vertical_acc': filtered_vertical[start_idx:end_idx],
                            'ap_acc': filtered_ap[start_idx:end_idx],
                            'ml_acc': filtered_ml[start_idx:end_idx],
                            'time': cycle_data['time'][start_idx:end_idx],
                            'subject_id': cycle_data['subject_id'],
                            'cycle_id': f"{cycle_data['cycle_id']}_{i}"
                        }
                        segmented_cycles.append(cycle_segment)
            
            segmented_data[subject_id] = segmented_cycles
            
        return segmented_data
    
    def normalize_cycles(self, segmented_data):
        """
        Normalize gait cycles to a standard length (100 data points)
        
        Args:
            segmented_data (dict): Dictionary with segmented gait cycles
            
        Returns:
            dict: Dictionary with normalized gait cycles
        """
        normalized_data = {}
        target_length = 100
        
        for subject_id, cycles in segmented_data.items():
            normalized_cycles = []
            
            for cycle in cycles:
                normalized_cycle = {}
                
                for signal_name in ['vertical_acc', 'ap_acc', 'ml_acc']:
                    original_signal = cycle[signal_name]
                    
                    # Interpolate to target length
                    original_indices = np.linspace(0, len(original_signal) - 1, len(original_signal))
                    target_indices = np.linspace(0, len(original_signal) - 1, target_length)
                    normalized_signal = np.interp(target_indices, original_indices, original_signal)
                    
                    normalized_cycle[signal_name] = normalized_signal
                
                # Keep other metadata
                normalized_cycle['subject_id'] = cycle['subject_id']
                normalized_cycle['cycle_id'] = cycle['cycle_id']
                normalized_cycle['time'] = np.linspace(0, 1, target_length)  # Normalized time
                
                normalized_cycles.append(normalized_cycle)
            
            normalized_data[subject_id] = normalized_cycles
            
        return normalized_data

def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    """
    preprocessor = GaitDataPreprocessor(sampling_rate=50)
    
    # Generate synthetic data
    raw_data = preprocessor.load_synthetic_data(n_subjects=30, n_samples_per_subject=50)
    
    # Segment gait cycles
    segmented_data = preprocessor.segment_gait_cycles(raw_data)
    
    # Normalize cycles
    normalized_data = preprocessor.normalize_cycles(segmented_data)
    
    return normalized_data

if __name__ == "__main__":
    # Example usage
    data = create_sample_dataset()
    print(f"Created dataset with {len(data)} subjects")
    print(f"Total cycles: {sum(len(cycles) for cycles in data.values())}")
