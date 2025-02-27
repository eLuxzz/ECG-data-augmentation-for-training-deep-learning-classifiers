import wfdb
import h5py
import numpy as np
import os
from tqdm import tqdm

# Paths
record_file = "RECORDS"  # Path to the record list
hdf5_filename = "output.hdf5"

# Read the list of records (without extensions)
with open(record_file, "r") as f:
    records = [line.strip() for line in f.readlines()]

# Number of records
num_records = len(records)

# Create HDF5 file
with h5py.File(hdf5_filename, "w") as hdf5_file:
    # Create dataset with shape (num_records, 5000, 12)
    dataset = hdf5_file.create_dataset("tracings", shape=(num_records, 5000, 13), dtype=np.float32)

    # Loop through each record in the file
    for i, record_path in enumerate(tqdm(records, desc="Processing records")):
        try:
            # Read the ECG record
            record = wfdb.rdrecord(record_path)  # WFDB will find .dat and .hea automatically
            signals = record.p_signal  # Shape (5000, 12)
            id = record_path.split("/")[2].replace("_hr", "")
            # Store in HDF5 file
            
            id_col = np.zeros((5000,1))
            id_col[0] = id
            

            new_col = np.hstack((id_col, signals))
            
            print(new_col)
            dataset[i] =  new_col
        except Exception as e:
            print(f"Error processing {record_path}: {e}")

print(f"Successfully saved {num_records} records to {hdf5_filename}")
