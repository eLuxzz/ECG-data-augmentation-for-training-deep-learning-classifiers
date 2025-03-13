import h5py
import pandas as pd

class Fileloader:
    def __init__(self):
        self.files = []
    """A dedicated class for loading in raw data from HDF5 and CSV files."""
    def getData(self, pathHDF5, setName, pathCSV = None):
        """Loads dataset, setName, from HDF5 file, and, if provided, csv data
        Args:
            pathHDF5 (str): Path to HDF5 file to load from
            setName (str): Dataset name in HDF5 file
            pathCSV (str, optional): Path to CSV, containing Annotations belonging to dataset.
        Returns:
            tuple:
                - hdf5File[setName] [any], Signal Data
                - annotationData [any], Annotation values
        """
        hdf5Data = self.__loadDataFromHDF5(pathHDF5, setName)
        annotationData = self.__loadDataFromCSV(pathCSV)
        return hdf5Data, annotationData

    def __loadDataFromHDF5(self,pathHDF5, setName):
        """Load data specified data set from HDF5 
        Args:
            pathHDF5 (str): Path to HDF5
            setName (string): Dataset name
        Returns:
            signals [float32]: An array containing all data signals
        """
        file = (h5py.File(pathHDF5, "r"))
        signals = file[setName]
        self.files.append(file)
        return signals

    def __loadDataFromCSV(self,pathCSV):
        """Loads data from csv file
        Args:
            pathCSV (str): Path to CSV file
        Returns:
            annotationData [any]: Csv values
        """
        if pathCSV is None:
            return None
        else:
            annotationData = pd.read_csv(pathCSV).values
        return annotationData
    def __del__(self):
        for file in self.files:
            file.close()    
