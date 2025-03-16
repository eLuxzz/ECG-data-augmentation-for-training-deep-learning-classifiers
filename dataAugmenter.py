import numpy as np
import tensorflow as tf
class DataAugmenter:
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, signalData,label, mean=0, std=0.05):
        """
        Adds Gaussian noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (batch, timesteps, channels)
        mean: float
            Mean of the Gaussian noise
        std: float
            Standard deviation of the Gaussian noise
        
        Returns:
        augmented_data: ndarray
            ECG data with added Gaussian noise
        """
        gaussian_noise = tf.random.normal(shape=tf.shape(signalData), mean=mean, stddev=std, dtype=tf.float16)
        augmented_data = signalData + gaussian_noise
        return augmented_data, label
    
    def add_powerline_noise(self, signalData,label, frequency=50, amplitude=0.01, sampling_rate=500):
        """
        Adds sinusoidal powerline noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (batch, timesteps, channels)
        frequency: float
            Frequency of powerline noise (default: 50Hz)
        amplitude: float
            Amplitude of the noise
        sampling_rate: int
            Sampling rate of the signal
        
        Returns:
        augmented_data: ndarray
            ECG data with added powerline noise
        """
        t = np.linspace(0, signalData.shape[1] / sampling_rate, signalData.shape[1], dtype=np.float16)
        
        powerline_noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)
        powerline_noise = powerline_noise[np.newaxis, :, np.newaxis]  

        
        augmented_data = signalData + powerline_noise
        return augmented_data, label
    
    # def add_baseline_wander(self, signalData, frequency=0.5, amplitude=0.1, sampling_rate=500):
    
    #     t = np.arange(signalData.shape[1], dtype=np.float16) / sampling_rate  
    #     baseline_noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)  
        
    #     baseline_noise = baseline_noise[np.newaxis, :, np.newaxis]  
    #     baseline_noise = np.tile(baseline_noise, (signalData.shape[0], 1, signalData.shape[2]))  
        
    #     augmented_data = signalData + baseline_noise
    #     return augmented_data
    def add_baseline_wander(self, signalData,label, frequency=0.5, amplitude=0.1, sampling_rate=500):
        """
        Adds baseline wander noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (batch, timesteps, channels)
        frequency: float
            Frequency of the baseline wander (default: 0.5Hz)
        amplitude: float
            Amplitude of the noise
        sampling_rate: int
            Sampling rate of the signal
        
        Returns:
        augmented_data: ndarray
            ECG data with added baseline wander noise
        """
        t = np.linspace(0, signalData.shape[1] / sampling_rate, signalData.shape[1], dtype=np.float16)
        noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)

        noise = noise.reshape(1, -1, 1)  # Expand dimensions for broadcasting
        return signalData + noise, label  # Returns a new array

