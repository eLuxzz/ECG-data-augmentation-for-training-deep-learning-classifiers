import numpy as np
import tensorflow as tf
import pandas as  pd


class DataAugmenter:
    def __init__(self, timesteps = 5000, leads=12):
        self.timesteps = timesteps
        self.n_leads = leads
        self.all_generated_arrays = []
        self.arrays = {} 
        pass
    @tf.function
    def add_gaussian_noise(self, signalData,label, mean=0, std=0.02):
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
    
    def add_powerline_noise(self, signalData,label, frequency=50, amplitude=0.15, sampling_rate=500):
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
        amplitude = tf.random.uniform((), 0.05, amplitude, dtype=tf.float32)
        t = np.linspace(0, signalData.shape[1] / sampling_rate, signalData.shape[1], dtype=np.float16)
        powerline_noise = amplitude * np.sin(2 * np.pi * frequency * t, dtype=np.float16)

        return signalData + powerline_noise, label
    
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
       
        # Choose a strong warp factor for noticeable changes
        warp_factor = tf.random.uniform([], minval=warp_factor_range[0], maxval=warp_factor_range[1], dtype=tf.float16)
        # Compute new length after warping
        warped_timesteps = tf.cast(tf.round(tf.cast(self.timesteps, tf.float16) * warp_factor), tf.int16)
        
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
        # Note for resize, returns dtype=float32 unless method is neareset
        signal = tf.image.resize(signal, [warped_timesteps, self.n_leads], method="nearest") # interpolate as an img
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