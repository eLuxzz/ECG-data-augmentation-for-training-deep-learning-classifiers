import numpy as np
import tensorflow as tf
import random
class DataAugmenter:
    def __init__(self, timesteps = 5000, leads=12):
        self.timesteps = timesteps
        self.n_leads = leads
        self.baseline = tf.Variable(tf.zeros((5000, 12)), trainable=False)
        self.all_generated_arrays = []
        self.arrays = {} 
        pass
    @tf.function
    def add_gaussian_noise(self, signalData,label, mean=0, std=0.007):
        """
        Adds Gaussian noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (timesteps, leads)
        mean: float
            Mean of the Gaussian noise
        std: float
            Standard deviation of the Gaussian noise
        
        Returns:
        augmented_data: ndarray
            ECG data with added Gaussian noise
        """
        std = tf.random.uniform((), 0.005, 0.008)
        gaussian_noise = tf.random.normal(shape=tf.shape(signalData), mean=mean, stddev=std, dtype=tf.float32)
        return signalData + gaussian_noise, label
    
    def add_powerline_noise(self, signalData,label, frequency=50, amplitude=0.01, sampling_rate=500):
        """
        Adds sinusoidal powerline noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (timesteps, leads)
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
        amplitude = tf.random.uniform((), 0.003, amplitude, dtype=tf.float32)
        t = np.linspace(0, signalData.shape[1] / sampling_rate, signalData.shape[1], dtype=np.float16)
        powerline_noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)

        return signalData + powerline_noise, label
    
    @tf.function
    def random_disturbance(self, signalData,label):
        """
        Adds random disturbance noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (timesteps, leads)
        label: ndarray
            The annotations belonging to the signal
        Returns:
        augmented_data: ndarray
            ECG data with random disturbance noise
        """
        #NOTE: Denna ger väldigt kantiga kurvor
        # Find the max value in each lead (column), and set a dynamic scaling factor as a percentage of the max value for each lead
        reduce_max = tf.reduce_max(tf.abs(signalData), axis=0)
        scaling = tf.random.uniform(signalData.shape, reduce_max*0.08, reduce_max*0.2, dtype=tf.float32)
        # Randomly decide whether to add or subtract the scaling
        sign = tf.random.uniform((), 0, 2, dtype=tf.int32)  # Random integer: 0 or 1
        # Apply the scaling based on the random choice (0 -> add, 1 -> subtract)
        return signalData + tf.where(sign == 0, scaling, -scaling), label
    
    
    @tf.function
    def time_warp_crop(self, signal, label, warp_factor_range=(0.85, 1.15)):
        
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
        with tf.device("/GPU:0"):
            # Choose a strong warp factor for noticeable changes
            warp_factor = tf.random.uniform([], minval=warp_factor_range[0], maxval=warp_factor_range[1], dtype=tf.float16)
            # Compute new length after warping
            warped_timesteps = tf.cast(tf.round(tf.cast(self.timesteps, tf.float16) * warp_factor), tf.int16)
            signal = tf.expand_dims(signal, axis=-1)  # Shape (5000, 12, 1)
            # Note for resize, returns dtype=float32 unless method is neareset
            signal = tf.image.resize(signal, [warped_timesteps, self.n_leads], method="nearest") # interpolate as an img
            signal = tf.squeeze(signal, axis=-1)  # Remove extra dim
            
            signal = tf.cond(
                warped_timesteps > self.timesteps,
                lambda: signal[:self.timesteps, :],  # Crop if too long
                lambda: tf.concat([signal, tf.repeat(signal[-1:, :], self.timesteps - warped_timesteps, axis=0)], axis=0)
                # lambda: tf.pad(signal, 
                #                [[0, self.timesteps - warped_timesteps], [0, 0]], 
                #                mode="CONSTANT", 
                #                constant_values=signal[-1, :])  # Efficient padding
            )
            signal.set_shape([self.timesteps, self.n_leads])  
        
        return signal, label    
    
    @tf.function
    def median_filter(self, signal, label, kernel_size=1001, addBaseline = True):
        """Apply a 1D median filter approximation using depthwise convolution."""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        with tf.device("/GPU:0"):  # Force execution on GPU
            signal = tf.expand_dims(signal, axis=0)
            signal = tf.expand_dims(signal, axis=-1)
            
            filter_weights = tf.ones([kernel_size, 1, 1,1]) /kernel_size

            # Apply depthwise Conv1D (efficient for GPUs)
            filtered_signal = tf.nn.conv2d(
                signal,
                filters=filter_weights,
                strides=[1,1,1,1],
                padding="SAME",
                data_format="NHWC"  # (Batch, Time, Channels)
            )
            # filtered_signal = tf.squeeze(filtered_signal, axis=-1)  # Shape (5000, 12)
            filtered_signal = tf.squeeze(filtered_signal, axis=0)  # Shape (5000, 12)
            filtered_signal = tf.squeeze(filtered_signal, axis=-1)  # Shape (5000, 12)
            signal = tf.squeeze(signal, axis=0)
            signal = tf.squeeze(signal, axis=-1)
            # Subtract the baseline wander
            signal = signal - filtered_signal

            if addBaseline and self.baseline is not None:
                signal = signal + self.baseline

            # Store baseline
            self.baseline.assign(filtered_signal)
        return signal, label
    
    def selective_amplitude_scaling(self,signal,label, lead_indices=[0, 1, 2, 5, 6]):
        # Generate random scale factors for selected leads (e.g., between 0.8 and 1.2)
        scale_factors = tf.random.normal(shape=[len(lead_indices)], mean=1, stddev=0.05, dtype=tf.float32)
        
        # Create a ones tensor (identity scaling for unselected leads)
        scaling_tensor = tf.ones([12], dtype=tf.float32)
        
        # Assign random scaling values to selected leads
        scaling_tensor = tf.tensor_scatter_nd_update(
            scaling_tensor, 
            indices=tf.constant([[i] for i in lead_indices], dtype=tf.int32), 
            updates=scale_factors
        )
        print(scaling_tensor)
        return signal * scaling_tensor, label
    
    def amplitude_scaling(self, signal, label):
        # if label[2] > 0.5: # Only scale V1-V6 for hyp.
        #     return self.selective_amplitude_scaling(signal, label)
        factor = tf.random.normal(shape=(1,12), mean=1, stddev=0.05, dtype=tf.float32)
        return signal * factor, label
    
    @tf.function
    def hga_kent(self, signal, label, segments=4, window_size=200):
        """
        Applies Host-Guest Augmentation to the ECG signal.

        Parameters:
        signal: Tensor of shape (timesteps, 12)
            Input ECG signal (expected shape: (5000, 12))
        label: Tensor
            Corresponding label.
        segments: int
            Number of segments to divide the signal into.
        window_size: int
            Size of the window around the segment boundary to search for low-activity points.

        Returns:
        Augmented ECG signal (same shape as input) and label.
        """
        with tf.device("/GPU:0"):
            # Calculate approximate segment length
            segment_length = self.timesteps // segments
            
            # Initialize a list to store the segments
            segments_list = []
            previous_end = 0
            
            for segment in range(segments):
                # Define the start and end of the search window
                start_idx = previous_end
                end_idx = start_idx + segment_length + window_size//2
                # end_idx = tf.reduce_min((self.timesteps, start_idx + segment_length + window_size//2))
                # tf.print("Start: ", start_idx, "End: ", end_idx)
                if segment == segments-1 or end_idx >= self.timesteps:
                    end_idx = self.timesteps
                else:
                    # Find the index within the window where activity is lowest
                    window = signal[end_idx - window_size:end_idx]
                    activity_levels = tf.reduce_mean(tf.abs(window), axis=1)  # Compute activity for each timestep
                    previous_end = end_idx = tf.argmin(activity_levels, output_type=tf.dtypes.int32) + end_idx - window_size
                # tf.print("Segment", segment,"Start: ", start_idx, "End: ", end_idx)
                # tf.print("After argmin: ", end_idx)
                # Extract the segment
                segments_list.append(signal[start_idx:end_idx])
            #%% Ev ta average på ett fönster av 100 och lägg till medel på hela segmentet för att matcha ihop ändarna
            # # Adjust the segments to match start and endpoints
            # # Define a window size for averaging
            # # Calculate the average of the last `window_size` points of the current segment
            # end_avg = tf.reduce_mean(segments_list[prev_index][-window_size:], axis=0)
            # # Calculate the average of the first `window_size` points of the next segment
            # start_avg = tf.reduce_mean(segments_list[i][:window_size], axis=0)
            # # Compute the adjustment needed to align the segments
            # adjustment = prev_avg - start_avg
            
            # # Adjust the next segment to align with the current segment
            # segments_list[i] += adjustment
                        
            # window_size = 100
            #%% A ugly code that works around tf shape issues with concat
            range_values = tf.range(segments)
            range_values = tf.random.shuffle(range_values)
            new_signal = tf.zeros((0,12)) # instantiation of new_signal
            for i in range(segments):
                if i == range_values[0]:
                    new_signal = segments_list[i] # Always set
            for j in range_values[1:]:
                for i in range(segments):
                    if i == j:
                        new_signal = tf.concat([new_signal, segments_list[i]], axis=0)
                
            new_signal.set_shape([self.timesteps, self.n_leads])
        return new_signal, label
       
        #%% An attempt to use proper tf code to concat the segments

        # range_values = list(range(3))
        # segments_tensor = tf.stack(segments_list) # Shape (3, 5000, 12)
        # range_values = tf.range(3)
        # # Shuffle the range
        # range_values = tf.random.shuffle(range_values)
        # shuffled_segments = tf.gather(segments_tensor, range_values, axis=0)
        # new_signal = shuffled_segments[0]
        # for i in range(1, len(shuffled_segments)):
        #     new_signal = tf.concat([new_signal, shuffled_segments[i]], axis=0)
        # new_signal.set_shape([self.timesteps, self.n_leads])
        # tf.print("Signal", tf.shape(new_signal))

        # return new_signal, label