import numpy as np
import tensorflow as tf
import pandas as  pd


class DataAugmenter:
    def __init__(self, timesteps = 5000, leads=12):
        self.timesteps = timesteps
        self.n_leads = leads
        self.all_generated_arrays = []
        self.arrays = {} 
        self.baseline = tf.Variable(tf.zeros((5000, 12)), trainable=False)
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
    
    def add_baseline_wander(self, signalData, label, frequency=0.5, amplitude=0.1, sampling_rate=500):
        """
        Adds baseline wander noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (timesteps, leads)
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
        amplitude = tf.random.uniform((), 0.05, amplitude, dtype=tf.float32)
        noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)
        return signalData + noise, label  # Returns a new array
    
    @tf.function
    def random_scaling(self, signalData,label):
        """
        Adds random scaling noise to the ECG signal.
        
        Parameters:
        signalData: ndarray
            Input ECG data with shape (timesteps, leads)
        label: ndarray
            The annotations belonging to the signal
        Returns:
        augmented_data: ndarray
            ECG data with added random scaling
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
    def add_time_warp(self, signal, label, warp_factor_range=(0.5, 1.5)):
        """
        Randomly stretches or compresses the ECG signal in time.

        Parameters:
        signalData: Tensor
            Input ECG data with shape (timesteps, leads)
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

        # Generate a random warp factor
        warp_factor = tf.random.uniform([], minval=warp_factor_range[0], maxval=warp_factor_range[1], dtype=tf.float16)
        # Compute new time indices
        warped_timesteps = tf.cast(tf.round(tf.cast(self.timesteps, tf.float16) * warp_factor), tf.int16)

        # Switch lead as prep for resize, as resize 
        # Lead 3 & 4
        # Lead 8 & 11
        switchIndex = [0,1,3,2,4,5,6,10,8,9,7,11]
        signal = tf.gather(signal, switchIndex, axis=1) 

        # Reshape signal for correct warping
        signal = tf.expand_dims(signal, axis=-1)  # Shape (5000, 12, 1)

        # Apply time warping with bilinear interpolation
        # warped_signal = tf.image.resize(signal_expanded, [num_leads, warped_timesteps], method="bilinear")
        signal = tf.image.resize(signal, [warped_timesteps, self.n_leads], method="lanczos3")

        # Restore shape to (5000, 12)
        signal = tf.image.resize(signal, [self.timesteps, self.n_leads], method="gaussian")  # Resize back
        signal = tf.squeeze(signal, axis=-1)  # Remove extra dim

        signal = tf.cast(signal, dtype=tf.float32) # Resize returns float32, so cast it to float16
        signal = tf.gather(signal, switchIndex, axis=1)

        return signal, label
    
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
    def median_filter_old(self, signal, label, kernel_size=1001):
        """Apply a 1D median filter to each lead of the ECG signal.
        
        Args:
            signal: Tensor of shape (5000, 12), representing the ECG signal.
            kernel_size: The window size of the median filter (should be odd).
        
        Returns:
            Filtered signal with baseline wander removed.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        with tf.device("/GPU:0"):  # Force execution on GPU
            # Pad signal to handle edge effects (padding is done symmetrically)
            pad_size = kernel_size // 2
            padded_signal = tf.pad(signal, [[pad_size, pad_size], [0, 0]], mode="SYMMETRIC")
            padded_signal = tf.expand_dims(padded_signal, axis=0)  # Shape (1, 5000, 12)
            
            
            # Apply median filter using a sliding window
            padded_signal = tf.map_fn(lambda lead: tf.nn.pool(
                input=tf.expand_dims(lead, axis=0), 
                window_shape=[kernel_size], 
                pooling_type='AVG',  # Approximate median using average pooling
                padding='VALID',
                strides=[1]
            )[0], padded_signal)  # Transpose so we apply along time
            
            padded_signal = tf.squeeze(padded_signal, 0)
            # Remove baseline wander by subtracting the median-filtered signal
            signal = signal - padded_signal
            # if self.baseline is not None:
                # add baseline wander from previous processed signal.
                # signal = signal + self.baseline
            # Store baseline.
            self.baseline.assign(padded_signal)
        return padded_signal, label
    
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
            signal, label = self.add_gaussian_noise(signal, label)
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
        if label[2] > 0.5: # Only scale V1-V6 for hyp.
            return self.selective_amplitude_scaling(signal, label)
        factor = tf.random.normal(shape=(1,12), mean=1, stddev=0.05, dtype=tf.float32)
        return signal * factor, label
    def host_guest_augmentation(self, signalData, label, level=5, tolerance=0.1):
            """
            Implementerar Host-Guest-principen för EKG-signalaugmentering.
            """
            # if label[2] > 0: # Gör inte augmentering på HYP?
            #     return signalData, label
            segment_length = self.timesteps // level
            signal_copy = tf.identity(signalData)
            
            for i in range(level - 1):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length
                
                host_segment = signalData[start_idx:end_idx, :]
                guest_segment = signalData[end_idx:(end_idx + segment_length), :] if end_idx + segment_length < self.timesteps else signalData[:segment_length, :]
                
                difference = tf.abs(host_segment - guest_segment)
                mask = difference > tolerance  # Identifiera skillnader
                
                modified_segment = tf.where(mask, guest_segment, host_segment)
                signal_copy = tf.tensor_scatter_nd_update(signal_copy, tf.range(start_idx, end_idx)[:, None], modified_segment)
            
            return signal_copy, label




    def Host_Guest_ECG_Augmentation_12_Leads(self, data,file_path,Lead_No,Level,threshold):  
        df = pd.DataFrame(columns=['Segment', 'Start', 'End'])
        split_value = round(len(data) / Level, ) #divide the ECG signal array into equal length of the level

        start = 0
        end = split_value  

        row = 0
        div_Segment = 1    #segment index starts from 1 i.e. not from 0

        #making the dataframe with start and end value of each segment
        for i in range(1, Level + 1):
            df.loc[row, 'Segment'] = round(div_Segment, )
            df.loc[row, 'Start'] = start
            df.loc[row, 'End'] = end

            start = (end+1)
            end += split_value

            if end > len(data):
                end = len(data)
            div_Segment+=1
            row+=1

        My_original_dataframe=df  #making copy of the dataframe
        current = 0 #the current dealing index of dataframe
        original_df = df.index.tolist()
        Max_Original_df_index=max(original_df)

        #starting a loop on dataframe index wise to decide about the Host/Guest segment in each segment give in dataframe
        for ind in df.index:
        
            # check if segment division is given 2 by the user then do not execute i.e. no segment shuffling is possible here
            if len(original_df)== 2 and max(original_df) == ind:
                
                do_not_execute=1

            elif max(original_df) == ind and len(original_df)>2:

                Start_value = df.iloc[ind]['Start']
                End_value = df.iloc[ind]['End']

                Guest_B = current
                Host_B = min(original_df)
                
                Host_A="Null"
                Guest_A="Null"
                
                idx = df.index.tolist()
                idx.pop(current)
                df = df.reindex([current] + idx)

                #calling modified tolerance method for pruning and joing segments
                self.Modified_measure_tolerance(df.index.tolist(),data,file_path,current,ind,Lead_No,Max_Original_df_index,
                                        Start_value,End_value,Guest_B,Host_B,Guest_A,Host_A,My_original_dataframe,threshold)

                #organizing the sequence when one segment is shuffled and stored in array for final step
                df = df.reindex(original_df)
                
            
            else:
                Start_value = df.iloc[ind]['Start']
                End_value = df.iloc[ind]['End']
                
                Guest_B = current
                Host_B = max(original_df)
                
                
                Host_A = (current - 1)
                Guest_A = (current + 1)
                
                if current == min(original_df) or current == max(original_df):
                    Host_A="Null"
                    Guest_A="Null"
                    
                idx = df.index.tolist()
                idx.pop(current)
                df = df.reindex(idx + [current])
                
                self.Modified_measure_tolerance(df.index.tolist(),data,file_path,current,ind,Lead_No,Max_Original_df_index,
                                        Start_value,End_value,Guest_B,Host_B,Guest_A,Host_A,My_original_dataframe,threshold)

                df = df.reindex(original_df)
                current += 1


    #Method for pruning the Host segment and re-joing        
    def Segment_Pruning(self, host_val, guest_val, My_original_dataframe, ind, threshold_value, dataframe_index_array,
                        my_final_numpy_array, Host_array, Host, Triggered_From_Which_Host):
        
        segments_shuffle_ready = 0
        delta_threshold = round((0.50*threshold_value),)
        tf.autograph.experimental.set_loop_options(
                shape_invariants=[(my_final_numpy_array, tf.TensorShape([None]))]
            )
        while segments_shuffle_ready == 0:

                    #calculating the mean difference between host/guest segment based on threshold_values and join if it is higher i.e. >= 1
                    #error = np.mean( host_val != guest_val )
                    some_arr = host_val != guest_val
                    some_arr = tf.cast(some_arr, tf.float32)
   
                    error =  tf.reduce_mean(some_arr)
                    if error >=  1.0:
                        Last_Segment_Ended_On = My_original_dataframe['End'][ind]
                        last_index = ind
                        segments_shuffle_ready=1
                        threshold_value=-(threshold_value)
                        
                        if Host == min (dataframe_index_array) and Triggered_From_Which_Host == "B":

                            my_final_numpy_array =tf.concat((my_final_numpy_array, Host_array[-(threshold_value):]), axis=-1)

                        else: 
                            my_final_numpy_array = tf.concat((my_final_numpy_array, Host_array[:threshold_value]), axis=-1)

                    elif  Host == min (dataframe_index_array) and Triggered_From_Which_Host == "A":
                            
                            Host_array=Host_array[delta_threshold:]
                            host_val=Host_array[:threshold_value]
                            segments_shuffle_ready=0

                    else: 

                        Host_array=Host_array[:-delta_threshold]
                        host_val=Host_array[-threshold_value:]

                        ###host_val=Host_array[-(threshold_value+round((0.50*threshold_value),)):-(delta_threshold)]
                        #########threshold_value=threshold_value+round((0.50*threshold_value),)

                        segments_shuffle_ready=0
        
        return my_final_numpy_array
                
    def Modified_measure_tolerance(self, dataframe_index_array, ecg_array, ecg_file_path, current, func_ind, Lead_No, Max_Original_df_index,
                                Start_value, End_value, Guest_B, Host_B, Guest_A, Host_A, My_original_dataframe, threshold):
        
        # Calculate threshold value for digit points
        total_length_of_the_segment = (End_value - Start_value)
        threshold_value = round((1 / threshold) * total_length_of_the_segment)

        # Initialize Host/Guest segments for Scenario A (if applicable)
        if Guest_A != "Null" and Host_A != "Null":
            Guest_A_array = ecg_array[My_original_dataframe.iloc[Guest_A]['Start']:My_original_dataframe.iloc[Guest_A]['End']]
            Host_A_array = ecg_array[My_original_dataframe.iloc[Host_A]['Start']:My_original_dataframe.iloc[Host_A]['End']]
            host_A_val = Host_A_array[-(threshold_value):]
            guest_A_val = Guest_A_array[:threshold_value]

        # Initialize Host/Guest segments for Scenario B
        Guest_array = ecg_array[My_original_dataframe.iloc[Guest_B]['Start']:My_original_dataframe.iloc[Guest_B]['End']]
        Host_array = ecg_array[My_original_dataframe.iloc[Host_B]['Start']:My_original_dataframe.iloc[Host_B]['End']]
        host_val = Host_array[-(threshold_value):]
        guest_val = Guest_array[:threshold_value]

        # Initialize variables for segment processing
        last_index = -1
        Is_First_index = 1
        my_final_numpy_array = []
        Last_Segment_Ended_On = 0
        segments_shuffle_ready = 0

        # Process each segment in dataframe_index_array
        for ind in dataframe_index_array:
            if Is_First_index == 1 and Host_A == "Null":
                my_final_numpy_array = tf.concat((my_final_numpy_array, ecg_array[My_original_dataframe['Start'][ind]:My_original_dataframe['End'][ind]]), axis=-1)
                Is_First_index = 0
                Last_Segment_Ended_On = My_original_dataframe['End'][ind]
                last_index = ind
            
            elif ind == Host_A:
                my_final_numpy_array = self.Segment_Pruning(host_A_val, guest_A_val, My_original_dataframe, ind, 
                                                threshold_value, dataframe_index_array, my_final_numpy_array, 
                                                Host_A_array, Host_A, "A")
            elif ind == Host_B:
                my_final_numpy_array = self.Segment_Pruning(host_val, guest_val, My_original_dataframe, ind, 
                                                threshold_value, dataframe_index_array, my_final_numpy_array, 
                                                Host_array, Host_B, "B")
            else:
                my_final_numpy_array = tf.concat((my_final_numpy_array, ecg_array[My_original_dataframe['Start'][ind]:My_original_dataframe['End'][ind]]), axis=-1)
                Last_Segment_Ended_On = My_original_dataframe['End'][ind]
                last_index = ind

    # Dictionary to store tensors for each func_ind


        if func_ind in self.arrays:
            ij=0

            for ii in my_final_numpy_array:
                self.arrays[func_ind]= tf.tensor_scatter_nd_update(self.arrays[func_ind], [[ij, Lead_No]], [ii])
                ij=ij+1
            my_final_numpy_array = []
        else:
            shape = (5000, 12)
            self.arrays[func_ind]= tf.zeros(shape)
            
            ij=0
            for ii in my_final_numpy_array:
                self.arrays[func_ind]= tf.tensor_scatter_nd_update(self.arrays[func_ind], [[ij, Lead_No]], [ii])
                #arrays[func_ind][ij][Lead_No]=ii
                ij=ij+1
            my_final_numpy_array = []

        #when all leads are finished starting from index 0 to 11 then store in a numpy file for futher ECG graph plotting.
        if Lead_No == 11: 

            myarray=self.arrays[func_ind]
            
            self.all_generated_arrays.append(myarray)
            my_final_numpy_array = []
            del self.arrays[func_ind]  #making the dynamic array null for next ecg signal

    def main(self, signal, label):

        #Required variables
        file = "C:\\Users\\amjad\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\records500\\21000\\21000_hr.npy"
        No_of_leads=12
        Level=2 #defines how many sample needs to be create from one original sample i.e., when 3 is given it will create 3 new from each original signal (Level must be greater than 2)
        threshold=100 #indicate the cutoff value to acquire the datapoints from host/guest segment 



        for Lead in range (0, No_of_leads,1): #initiating 12 times loop on variable x to extract lead/column wise data (which then passed to Host_Guest_ECG_Augmentation_12_Leads method)
            n=signal[:,Lead]
            # n=np.array(n).flatten()
            self.Host_Guest_ECG_Augmentation_12_Leads(n,file,Lead,Level,threshold)
        #print(self.all_generated_arrays[0])
        return self.all_generated_arrays[0], label