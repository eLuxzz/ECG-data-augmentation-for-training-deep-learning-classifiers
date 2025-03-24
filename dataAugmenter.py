import numpy as np
import tensorflow as tf
class DataAugmenter:
    def __init__(self, timesteps = 5000, leads=12):
        self.timesteps = timesteps
        self.leads = leads
        pass
    
    def add_gaussian_noise(self, signalData,label, mean=0, std=0.02):
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
        gaussian_noise = tf.random.normal(shape=tf.shape(signalData), mean=mean, stddev=std, dtype=tf.float32)
        return signalData + gaussian_noise, label
    
    def add_powerline_noise(self, signalData,label, frequency=50, amplitude=0.1, sampling_rate=500):
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
        
        return signalData + powerline_noise, label
    
    def add_baseline_wander(self, signalData, label, frequency=0.5, amplitude=0.1, sampling_rate=500):
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

        return signalData + noise, label  # Returns a new array
    def add_time_warp(self, signal, label, warp_factor_range=(0.5, 1.5)):
        """
        Randomly stretches or compresses the ECG signal in time.

        Parameters:
        signalData: Tensor
            Input ECG data with shape (timesteps, channels)
        label: Tensor
            Corresponding label
        warp_factor_range: tuple
            Range of stretching/compression factors (e.g., (0.8, 1.2) means 80%-120% stretch)

        Returns:
        Warped ECG signal and label
        """
        # Generate a random warp factor within the range
        # Get the original shape
        
        # Get the original shape
        # timesteps = tf.shape(signal)[0]  # Expected to be 5000
        # num_leads = tf.shape(signal)[1]  # Expected to be 12

        timesteps = self.timesteps
        num_leads = self.leads


        # Generate a random warp factor
        warp_factor = tf.random.uniform([], minval=warp_factor_range[0], maxval=warp_factor_range[1])
        # Compute new time indices
        warped_timesteps = tf.cast(tf.round(tf.cast(timesteps, tf.float32) * warp_factor), tf.int32)

        # Switch lead as prep for resize, as resize 
        # Lead 3 & 4
        # Lead 8 & 11
        switchIndex = [0,1,3,2,4,5,6,10,8,9,7,11]
        signal = tf.gather(signal, switchIndex, axis=1) 

        # Reshape signal for correct warping
        # signal_expanded = tf.transpose(signal, [1, 0])  # Change to (12, 5000)
        signal = tf.expand_dims(signal, axis=-1)  # Shape (12, 5000, 1)

        # Apply time warping with bilinear interpolation
        # warped_signal = tf.image.resize(signal_expanded, [num_leads, warped_timesteps], method="bilinear")
        signal = tf.image.resize(signal, [warped_timesteps, num_leads], method="lanczos3")

        # Restore shape to (5000, 12)
        signal = tf.image.resize(signal, [timesteps, num_leads], method="gaussian")  # Resize back
        signal = tf.squeeze(signal, axis=-1)  # Remove extra dim
        # warped_signal = tf.transpose(warped_signal, [1, 0])  # Back to (5000, 12)

        signal = tf.gather(signal, switchIndex, axis=1)

        return signal, label
    
    def time_warp_crop(self, signal, label, warp_factor_range=(0.85, 1.5)):
        """
        Applies time warping to a 12-lead ECG signal and crops instead of resizing.

        Parameters:
        signal: Tensor of shape (timesteps, 12)
            Input ECG signal (expected shape: (5000, 12))
        warp_factor_range: tuple
            Range of stretching/compression factors

        Returns:
        Warped and cropped ECG signal (same shape as input)
        """
        # timesteps = tf.shape(signal)[0]  # 5000
        # num_leads = tf.shape(signal)[1]  # 12
        timesteps = self.timesteps
        num_leads = self.leads
        # Choose a strong warp factor for noticeable changes
        warp_factor = tf.random.uniform([], minval=warp_factor_range[0], maxval=warp_factor_range[1])
        # Compute new length after warping
        warped_timesteps = tf.cast(tf.round(tf.cast(timesteps, tf.float32) * warp_factor), tf.int32)
        #-------------
        # Generate new time indices for interpolation
        # original_indices = tf.linspace(0.0, tf.cast(timesteps - 1, tf.float32), timesteps)
        # warped_indices = tf.linspace(0.0, tf.cast(timesteps - 1, tf.float32), warped_timesteps)
        
        # Interpolate each lead separately using `tf.map_fn`
        # def interp_lead(lead_signal):
        #     interpolation = tf.numpy_function(np.interp, [warped_indices, original_indices, lead_signal], tf.float64)
        #     return tf.cast(interpolation, tf.float32)
        #-------------
        # warped_signal = tf.map_fn(interp_lead, tf.transpose(signal), dtype=tf.float32)
        # warped_signal = tf.transpose(warped_signal)  # Convert back to (timesteps, 12)

        signal = tf.expand_dims(signal, axis=-1)  # Shape (5000, 12, 1)
        signal = tf.image.resize(signal, [warped_timesteps, num_leads], method="bilinear") # interpolate as an img
        signal = tf.squeeze(signal, axis=-1)  # Remove extra dim

        # # **Instead of resizing, crop to original length**
        # if warped_timesteps > timesteps:
        #     # If stretched, take only the first 5000 samples
        #     signal = signal[:timesteps, :]
        # else:
        #     pad_size = timesteps - warped_timesteps
        #     # Repeat the last row of the signal to fill in the padding
        #     pad_values = signal[-1:, :]  # Get the last row

        #     # Concatenate the pad values to the end of the signal
        #     signal = tf.concat([signal, tf.repeat(pad_values, pad_size, axis=0)], axis=0)
        #     # pad_values = tf.tile(warped_signal[-1:, :], [pad_size, 1])  # Repeat last row
        #     # # pad_values = warped_signal[:pad_size, :]  # Copy first part of the signal
        #     # # Repeat the first part of the signal to fill in the padding
        #     # warped_signal = tf.concat([warped_signal, pad_values], axis=0)
        # signal = tf.ensure_shape(signal, [timesteps, num_leads]) # tf thingy, as the variable timesteps, num:leads is a symbolic value, and this requires constant int

        signal = tf.cond(
        warped_timesteps > timesteps,
        lambda: signal[:timesteps, :],  # Crop if too long
        lambda: tf.concat([signal, tf.repeat(signal[-1:, :], timesteps - warped_timesteps, axis=0)], axis=0)
        # lambda: tf.pad(signal, [[0, timesteps - warped_timesteps], [0, 0]], mode="CONSTANT")  # Efficient padding
    )
        signal.set_shape([timesteps, num_leads])  
        return signal, label    

