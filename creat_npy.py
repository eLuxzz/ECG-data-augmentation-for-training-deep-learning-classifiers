import os
import glob
import numpy as np
import wfdb

# Ange sökvägen där dina .dat och .hea-filer finns
dataset_path = "C:\\Users\\amjad\\Downloads\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\records500\\00000\\00001_hr"

# Hitta alla .hea-filer (de pekar på motsvarande .dat-filer)
hea_files = glob.glob(os.path.join(dataset_path, "*.hea"))

#for hea_file in hea_files:
    # Filnamnet utan extension
#record_name = os.path.splitext(os.path.basename(hea_file))[0]
#record_path = os.path.join(dataset_path, record_name)

# Läs signalen med wfdb
record = wfdb.rdsamp(dataset_path)

# Extrahera signaldata (första elementet i tuple)
signal_data = record[0]  

# Spara som .npy
npy_filename = f"{dataset_path}.npy"
np.save(npy_filename, signal_data)

print(f"Konverterade {dataset_path} till {npy_filename}")
