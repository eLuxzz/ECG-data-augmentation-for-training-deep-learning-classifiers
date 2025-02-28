import wfdb
import h5py
import numpy as np
import os
from tqdm import tqdm

# Paths
path_record_train = "data/PTB_XL_data/RECORDS_TRAIN.txt"  # Path to the record list
path_record_test = "data/PTB_XL_data/RECORDS_TEST.txt"
path_record_valid = "data/PTB_XL_data/RECORDS_VALID.txt"

path_hdf5_train = "data/PTB_XL_data/signals_train.hdf5"
path_hdf5_valid = "data/PTB_XL_data/signals_valid.hdf5"
path_hdf5_test = "data/PTB_XL_data/signals_test.hdf5"


def CreateHDF5(recordsPath, outputPath):

    # Read the list of records (without extensions)
    with open(recordsPath, "r") as f:
        records = [line.strip() for line in f.readlines()]

    # Number of records
    num_records = len(records)

    # Create HDF5 file
    with h5py.File(outputPath, "w") as hdf5_file:
        # Create dataset with shape (num_records, 5000, 12)
        dataset = hdf5_file.create_dataset("tracings", shape=(num_records, 5000, 12), dtype=np.float32)

        # Loop through each record in the file
        for i, path in enumerate(tqdm(records, desc="Processing records")):
            try:
                # Read the ECG record
                record = wfdb.rdrecord(path)  # WFDB will find .dat and .hea automatically
                signals = record.p_signal  # Shape (5000, 12)
                # Store in HDF5 file
                
                # Adds id as a column
                # id = path.split("/")[2].replace("_hr", "")
                # id_col = np.zeros((5000,1))
                # id_col[0] = id
                # new_col = np.hstack((id_col, signals))
                
                
                dataset[i] =  signals
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"Successfully saved {num_records} records to {path_hdf5_train}")
CreateHDF5(path_record_test, path_hdf5_test)
CreateHDF5(path_record_train, path_hdf5_train)
CreateHDF5(path_record_valid, path_hdf5_valid)