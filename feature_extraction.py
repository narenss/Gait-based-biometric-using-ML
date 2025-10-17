"""
Gait Feature Extraction Module
Extracts comprehensive features from gait data for biometric recognition
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class GaitFeatureExtractor:
    """
    Comprehensive feature extraction for gait biometric recognition
    """
    
    def __init__(self, sampling_rate=50):
        """
        Initialize the feature extractor
        
        Args:
            sampling_rate (int): Sampling rate of the gait data
        """
        self.sampling_rate = sampling_rate
        
    def extract_temporal_features(self, signal_data):
        """
        Extract temporal domain features from gait signals
        
        Args:
            signal_data (dict): Dictionary containing gait signals
            
        Returns:
            dict: Dictionary of temporal features
        """
        features = {}
        
        for signal_name, signal in signal_data.items():
            if signal_name in ['vertical_acc', 'ap_acc', 'ml_acc']:
                # Basic statistical features
                features[f'{signal_name}_mean'] = np.mean(signal)
                features[f'{signal_name}_std'] = np.std(signal)
                features[f'{signal_name}_var'] = np.var(signal)
                features[f'{signal_name}_skew'] = stats.skew(signal)
                features[f'{signal_name}_kurtosis'] = stats.kurtosis(signal)
                features[f'{signal_name}_rms'] = np.sqrt(np.mean(signal**2))
                features[f'{signal_name}_range'] = np.max(signal) - np.min(signal)
                
                # Peak features
                peaks = self._find_peaks(signal)
                features[f'{signal_name}_peak_count'] = len(peaks)
                features[f'{signal_name}_peak_mean'] = np.mean(signal[peaks]) if len(peaks) > 0 else 0
                
                # Zero crossing rate
                zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
                features[f'{signal_name}_zero_crossing_rate'] = len(zero_crossings) / len(signal)
                
                # Signal energy
                features[f'{signal_name}_energy'] = np.sum(signal**2)
                
                # Entropy
                features[f'{signal_name}_entropy'] = self._calculate_entropy(signal)
                
        return features
    
    def extract_frequency_features(self, signal_data):
        """
        Extract frequency domain features from gait signals
        
        Args:
            signal_data (dict): Dictionary containing gait signals
            
        Returns:
            dict: Dictionary of frequency features
        """
        features = {}
        
        for signal_name, signal in signal_data.items():
            if signal_name in ['vertical_acc', 'ap_acc', 'ml_acc']:
                # FFT analysis
                fft_signal = fft(signal)
                freqs = fftfreq(len(signal), 1/self.sampling_rate)
                magnitude = np.abs(fft_signal)
                power = magnitude**2
                
                # Frequency features
                features[f'{signal_name}_dominant_freq'] = freqs[np.argmax(magnitude[1:len(magnitude)//2]) + 1]
                features[f'{signal_name}_spectral_centroid'] = np.sum(freqs[:len(freqs)//2] * power[:len(power)//2]) / np.sum(power[:len(power)//2])
                features[f'{signal_name}_spectral_bandwidth'] = np.sqrt(np.sum(((freqs[:len(freqs)//2] - features[f'{signal_name}_spectral_centroid'])**2) * power[:len(power)//2]) / np.sum(power[:len(power)//2]))
                
                # Power spectral density
                f_psd, psd = welch(signal, fs=self.sampling_rate, nperseg=min(len(signal)//4, 64))
                features[f'{signal_name}_total_power'] = np.trapz(psd, f_psd)
                features[f'{signal_name}_peak_power_freq'] = f_psd[np.argmax(psd)]
                
                # Spectral features
                features[f'{signal_name}_spectral_rolloff'] = self._calculate_spectral_rolloff(psd, f_psd)
                features[f'{signal_name}_spectral_flux'] = self._calculate_spectral_flux(psd)
                
        return features
    
    def extract_waveform_features(self, signal_data):
        """
        Extract waveform-specific features for gait analysis
        
        Args:
            signal_data (dict): Dictionary containing gait signals
            
        Returns:
            dict: Dictionary of waveform features
        """
        features = {}
        
        for signal_name, signal in signal_data.items():
            if signal_name in ['vertical_acc', 'ap_acc', 'ml_acc']:
                # Gait cycle specific features
                if signal_name == 'vertical_acc':
                    # Double support phase detection (lowest points)
                    valleys = self._find_peaks(-signal)
                    features['double_support_ratio'] = len(valleys) / len(signal)
                    
                    # Heel strike detection
                    heel_strikes = self._find_peaks(signal)
                    features['heel_strike_count'] = len(heel_strikes)
                    
                # Symmetry features
                first_half = signal[:len(signal)//2]
                second_half = signal[len(signal)//2:]
                features[f'{signal_name}_symmetry'] = 1 - np.abs(np.mean(first_half) - np.mean(second_half)) / (np.abs(np.mean(first_half)) + np.abs(np.mean(second_half)) + 1e-8)
                
                # Regularity features
                features[f'{signal_name}_regularity'] = self._calculate_regularity(signal)
                
                # Smoothness features
                features[f'{signal_name}_smoothness'] = 1 / (1 + np.var(np.diff(signal)))
                
        return features
    
    def extract_phase_features(self, signal_data):
        """
        Extract gait phase-specific features
        
        Args:
            signal_data (dict): Dictionary containing gait signals
            
        Returns:
            dict: Dictionary of phase features
        """
        features = {}
        
        # Detect gait phases using vertical acceleration
        vertical_acc = signal_data.get('vertical_acc', np.zeros(100))
        
        # Find heel strikes (peaks in vertical acceleration)
        heel_strikes = self._find_peaks(vertical_acc)
        
        if len(heel_strikes) >= 2:
            # Calculate stance and swing phases
            stance_phase = (heel_strikes[1] - heel_strikes[0]) / len(vertical_acc)
            stance_phase = float(stance_phase) if hasattr(stance_phase, '__len__') else stance_phase
            
            features['stance_phase_ratio'] = stance_phase
            features['swing_phase_ratio'] = 1 - stance_phase
            
            # Double support phase (approximate)
            features['double_support_phase'] = min(0.2, stance_phase * 0.3)  # Typically 10-20% of cycle
            
            # Single support phase
            features['single_support_phase'] = stance_phase - features['double_support_phase']
        else:
            # Default values if heel strikes not detected properly
            features['stance_phase_ratio'] = 0.6
            features['swing_phase_ratio'] = 0.4
            features['double_support_phase'] = 0.15
            features['single_support_phase'] = 0.45
        
        return features
    
    def extract_all_features(self, signal_data):
        """
        Extract all features from gait signal data
        
        Args:
            signal_data (dict): Dictionary containing gait signals
            
        Returns:
            dict: Complete feature dictionary
        """
        all_features = {}
        
        # Extract different types of features
        all_features.update(self.extract_temporal_features(signal_data))
        all_features.update(self.extract_frequency_features(signal_data))
        all_features.update(self.extract_waveform_features(signal_data))
        all_features.update(self.extract_phase_features(signal_data))
        
        return all_features
    
    def extract_features_from_dataset(self, dataset):
        """
        Extract features from entire dataset
        
        Args:
            dataset (dict): Dictionary containing gait data for all subjects
            
        Returns:
            tuple: (features_df, labels) where features_df is DataFrame of features and labels is array of subject IDs
        """
        all_features = []
        labels = []
        
        for subject_id, cycles in dataset.items():
            subject_label = int(subject_id.split('_')[1])  # Extract subject number
            
            for cycle in cycles:
                # Extract features for this cycle
                signal_data = {
                    'vertical_acc': cycle['vertical_acc'],
                    'ap_acc': cycle['ap_acc'],
                    'ml_acc': cycle['ml_acc']
                }
                
                features = self.extract_all_features(signal_data)
                all_features.append(features)
                labels.append(subject_label)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        return features_df, np.array(labels)
    
    def _find_peaks(self, signal, height_threshold=0.1):
        """
        Find peaks in signal
        
        Args:
            signal (numpy.ndarray): Input signal
            height_threshold (float): Minimum height for peak detection
            
        Returns:
            numpy.ndarray: Peak indices
        """
        from scipy.signal import find_peaks
        height = np.max(signal) * height_threshold
        peaks, _ = find_peaks(signal, height=height, distance=5)
        return peaks
    
    def _calculate_entropy(self, signal):
        """
        Calculate Shannon entropy of signal
        
        Args:
            signal (numpy.ndarray): Input signal
            
        Returns:
            float: Shannon entropy
        """
        # Discretize signal
        hist, _ = np.histogram(signal, bins=20)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities
        entropy = -np.sum(prob * np.log2(prob))
        return entropy
    
    def _calculate_spectral_rolloff(self, psd, freqs, rolloff_threshold=0.85):
        """
        Calculate spectral rolloff frequency
        
        Args:
            psd (numpy.ndarray): Power spectral density
            freqs (numpy.ndarray): Frequency bins
            rolloff_threshold (float): Rolloff threshold (default 85%)
            
        Returns:
            float: Spectral rolloff frequency
        """
        cumsum_psd = np.cumsum(psd)
        rolloff_index = np.where(cumsum_psd >= rolloff_threshold * cumsum_psd[-1])[0]
        return freqs[rolloff_index[0]] if len(rolloff_index) > 0 else freqs[-1]
    
    def _calculate_spectral_flux(self, psd):
        """
        Calculate spectral flux
        
        Args:
            psd (numpy.ndarray): Power spectral density
            
        Returns:
            float: Spectral flux
        """
        if len(psd) < 2:
            return 0
        flux = np.sum(np.diff(psd)**2)
        return flux
    
    def _calculate_regularity(self, signal):
        """
        Calculate signal regularity using coefficient of variation
        
        Args:
            signal (numpy.ndarray): Input signal
            
        Returns:
            float: Regularity measure (1/CV)
        """
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if mean_val == 0:
            return 0
        cv = std_val / abs(mean_val)
        return 1 / (1 + cv)

def create_feature_dataset():
    """
    Create a sample feature dataset for demonstration
    """
    from data_preprocessing import create_sample_dataset
    
    # Load sample data
    dataset = create_sample_dataset()
    
    # Extract features
    extractor = GaitFeatureExtractor()
    features_df, labels = extractor.extract_features_from_dataset(dataset)
    
    return features_df, labels

if __name__ == "__main__":
    # Example usage
    features_df, labels = create_feature_dataset()
    print(f"Extracted {features_df.shape[1]} features from {len(labels)} gait cycles")
    print(f"Features shape: {features_df.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique subjects: {len(np.unique(labels))}")
