import numpy as np
import glob
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dataPlotter import DataPlotter



#declaring an empty 1000 rows and 12 columns array to store 12-leads ecg signal
shape = (5000, 12)
Multi_Dim_Array= np.zeros(shape)

#Method for preparing the dataframe and deciding about Host/Guest for scenarios A & B.
def Host_Guest_ECG_Augmentation_12_Leads(data,file_path,Lead_No,Level,threshold):  
    
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
            #print(df)
            df = df.reindex([current] + idx)
            #print(df)

            #calling modified tolerance method for pruning and joing segments
            Modified_measure_tolerance(df.index.tolist(),data,file_path,current,ind,Lead_No,Max_Original_df_index,
                                       Start_value,End_value,Guest_B,Host_B,Guest_A,Host_A,My_original_dataframe,threshold)
            
            #organizing the sequence when one segment is shuffled and stored in array for final step
            df = df.reindex(original_df)
            #print(df)
            
        
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
            print(df)
            df = df.reindex(idx + [current])
            print(df)
            Modified_measure_tolerance(df.index.tolist(),data,file_path,current,ind,Lead_No,Max_Original_df_index,
                                       Start_value,End_value,Guest_B,Host_B,Guest_A,Host_A,My_original_dataframe,threshold)
            df = df.reindex(original_df)
            print(df)
            current += 1
            

#Method for pruning the Host segment and re-joing            
def Segment_Pruning(host_val, guest_val, My_original_dataframe, ind, threshold_value, dataframe_index_array,
                    my_final_numpy_array, Host_array, Host, Triggered_From_Which_Host):
    
    segments_shuffle_ready = 0
    delta_threshold = round((0.50*threshold_value),)
    while segments_shuffle_ready == 0:

                #calculating the mean difference between host/guest segment based on threshold_values and join if it is higher i.e. >= 1
                error = np.mean( host_val != guest_val )   
                if error >= 1.0:
                    Last_Segment_Ended_On = My_original_dataframe['End'][ind]
                    last_index = ind
                    segments_shuffle_ready=1
                    threshold_value=-(threshold_value)
                    
                    if Host == min (dataframe_index_array) and Triggered_From_Which_Host == "B":

                        my_final_numpy_array = np.append(my_final_numpy_array, Host_array[-(threshold_value):])

                    else: 
                        my_final_numpy_array = np.append(my_final_numpy_array, Host_array[:threshold_value])

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
            

#Method for calculating the tolerance value and preparing the data points of guest/host for first iteration (which then pass to Segment_Pruning method)
def Modified_measure_tolerance(dataframe_index_array,ecg_array,ecg_file_path,current,func_ind,Lead_No,Max_Original_df_index,
                               Start_value,End_value,Guest_B,Host_B,Guest_A,Host_A,My_original_dataframe,threshold):

    #calculating the threshold_value for digit points to be acquired from host segment for both scenario A and B
    total_length_of_the_segment=(End_value-Start_value)

    threshold_value=round((1/threshold)*total_length_of_the_segment)

    if Guest_A != "Null" and Host_A != "Null":
        
        #acquiring the full digit points of Guest/Host array
        Guest_A_array=ecg_array[My_original_dataframe.iloc[Guest_A]['Start']:My_original_dataframe.iloc[Guest_A]['End']]
        Host_A_array=ecg_array[My_original_dataframe.iloc[Host_A]['Start']:My_original_dataframe.iloc[Host_A]['End']]
        
        #applying threshold on the above given
        host_A_val=Host_A_array[-(threshold_value):]
        guest_A_val=Guest_A_array[:threshold_value]
        
    #Same threshold scenario for scenario B Host/Guest
    Guest_array=ecg_array[My_original_dataframe.iloc[Guest_B]['Start']:My_original_dataframe.iloc[Guest_B]['End']]
    Host_array=ecg_array[My_original_dataframe.iloc[Host_B]['Start']:My_original_dataframe.iloc[Host_B]['End']]
    
    host_val=Host_array[-(threshold_value):]
    guest_val=Guest_array[:threshold_value]
    

    #Setting the index and array values for Host segment pruning and then segment re-join
    last_index = -1
    Is_First_index=1
    my_final_numpy_array=[]
    Last_Segment_Ended_On=0
    segments_shuffle_ready = 0

    #Initiating loop and calling a new method of Segment_pruning based on scenarios i.e., both scenarios A & B have different Host sides to be pruned (left/right).
    for ind in dataframe_index_array:
        
        if Is_First_index == 1 and Host_A == "Null":
            my_final_numpy_array = np.append(my_final_numpy_array, ecg_array[My_original_dataframe['Start'][ind]:My_original_dataframe['End'][ind]])
            Is_First_index=0
            Last_Segment_Ended_On = My_original_dataframe['End'][ind]
            last_index = ind
        
        elif ind == Host_A:
            
            my_final_numpy_array = Segment_Pruning(host_A_val, guest_A_val, My_original_dataframe, ind, 
                                                   threshold_value, dataframe_index_array, my_final_numpy_array, 
                                                   Host_A_array,Host_A,"A")
        elif ind == Host_B:
            
            my_final_numpy_array = Segment_Pruning(host_val, guest_val, My_original_dataframe, ind, 
                                                   threshold_value, dataframe_index_array, my_final_numpy_array, 
                                                   Host_array,Host_B,"B")
        else:
            
            my_final_numpy_array = np.append(my_final_numpy_array, ecg_array[My_original_dataframe['Start'][ind]:My_original_dataframe['End'][ind]])
            Last_Segment_Ended_On = My_original_dataframe['End'][ind]
            last_index = ind     

    #extracting name from the path i.e., newly built signal needs a name which is based on levels such as 1,2,3 or so on. (OriginalFileName_Level)
    existing_file_name_1 = Path(ecg_file_path).stem
    new_file_name_1 = existing_file_name_1[:5]


    #Storing the Leads 1 to 12 newly built signal data into empty numpy array in column wise (1 to 12) which is a standard. 
    #Assuming the each column as 1000 values length which found in orignal dataset
    if f"Multi_Dim_Array_{func_ind}" in globals():
        ij=0

        for ii in my_final_numpy_array:
            globals()[f"Multi_Dim_Array_{func_ind}"][ij][Lead_No]=ii
            ij=ij+1
        my_final_numpy_array = []
    else:
        shape = (5000, 12)
        globals()[f"Multi_Dim_Array_{func_ind}"]= np.zeros(shape)
        
        ij=0
        for ii in my_final_numpy_array:
            globals()[f"Multi_Dim_Array_{func_ind}"][ij][Lead_No]=ii
            ij=ij+1
        my_final_numpy_array = []

    #when all leads are finished starting from index 0 to 11 then store in a numpy file for futher ECG graph plotting.
    if Lead_No == 11: 

        myarray=globals()[f"Multi_Dim_Array_{func_ind}"]
        
        np.save('C:\\Users\\amjad\\Desktop\\out\\' + new_file_name_1+"_" + str(current) + 'HEHEHH'+ '.npy', myarray)  # saving record
        my_final_numpy_array = []
        del(globals()[f"Multi_Dim_Array_{func_ind}"])  #making the dynamic array null for next ecg signal

    
        ##########threshold_value=round((threshold/100)*total_length_of_the_segment)

def main(x):

    #Required variables
    file = "C:\\Users\\amjad\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\records500\\21000\\21000_hr.npy"
    No_of_leads=12
    Level=3 #defines how many sample needs to be create from one original sample i.e., when 3 is given it will create 3 new from each original signal (Level must be greater than 2)
    threshold=1000 #indicate the cutoff value to acquire the datapoints from host/guest segment 

    #for file in glob.iglob(dataset_path + '**/*.npy', recursive=True):

    #x = np.load(file) #load the selected file data array into variable x

    for Lead in range (0, No_of_leads,1): #initiating 12 times loop on variable x to extract lead/column wise data (which then passed to Host_Guest_ECG_Augmentation_12_Leads method)
        n=x[:,[Lead]]
        n=np.array(n).flatten()
        Host_Guest_ECG_Augmentation_12_Leads(n,file,Lead,Level,threshold)


     
amjad  = np.load("C:\\Users\\amjad\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\records500\\00000\\00001_hr.npy") 
main(amjad)
x = np.load("C:\\Users\\amjad\\Desktop\\out\\21000_1HEHEHH.npy") 
# plt.figure(figsize=(10, 5))
# plt.plot(amjad[:,3], label="Base ", alpha=0.7)
# plt.plot(x[:,3], label="Aug ", alpha=0.7)
# plt.legend()
# plt.grid()
# plt.show()

# from  dataPlotter import DataPlotter

plotter = DataPlotter(
    amjad,
    x,
    same_graf=False,
    #sample_range=[900,1200],
    sample_range=[0,5000]
)

plotter.plot()
