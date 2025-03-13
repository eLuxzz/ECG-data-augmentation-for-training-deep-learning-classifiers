from fileloader import Fileloader
from dataAugmenter import DataAugmenter
import pdb

class Dataloader():
    def __init__(self, pathHDF5_training, pathCSV_training, pathHDF5_valid, pathCSV_valid, setNameTraining = "tracings", setNameValid="tracings"):
        """Set up dataloader with pathing, and instantiate needed help classes
        Args:
            pathHDF5_training (str): Path to HDF5 file containing training data
            pathCSV_training (str): Path to CSV file containing annotations related to training data
            pathHDF5_valid (str): Path to HDF5 file containing validation data
            pathCSV_valid (str): Path to CSV file containing annotations related to training data
            setNameTraining (str, optional): Dataset name in HDF5 File containing training data. Defaults to "tracings".
            setNameValid (str, optional): Dataset name in HDF5 File containing validation data. Defaults to "tracings".
        """
        self.PathHDF5_training = pathHDF5_training
        self.PathCSV_training = pathCSV_training
        self.PathHDF5_valid = pathHDF5_valid

        self.PathCSV_valid = pathCSV_valid
        self.SetName_Training = setNameTraining
        self.SetName_Valid = setNameValid

        self._DA = DataAugmenter()
        self._Fileloader = Fileloader()
        
    def getBaseTrainingData(self):
        """Loads and returns data without augmentation
        Returns:
            tuple:
                - signalData_training [any]: Base training data
                - annotationData_training [any]: Annotation data
        """
        signalData_training, annotationData_training = self._Fileloader.getData(self.PathHDF5_training, self.SetName_Training, self.PathCSV_training)
        return signalData_training, annotationData_training
    def getValidationData(self):
        """Loads and return validation data

        Returns:
            signalData_validation: Signal Data for validation
            annoatationData_validation: Annotation Data
        """
        signalData_validation, annotationData_validation = self._Fileloader.getData(self.PathHDF5_valid,self.SetName_Valid, self.PathCSV_valid)
        return signalData_validation, annotationData_validation
    def getAugmentedData(self, DAMethods):
        """Augmentate base data in the order the methods is provided.
        Args:
            DAMethods ([str]): Array containing DA method names, the order the DA is applied
        Returns:
            tuple:
                - augmentedData [any]: Augmentated training data
                - annotationData [any] : Annotation data
        """
        augmentedData, annotationData = self.getBaseTrainingData()
        print("Base-data loaded")
        for DAMethod in DAMethods:
            func = getattr(self._DA, DAMethod)
            if func:
                print(f"Applying {DAMethod}")
                augmentedData = func(augmentedData)
                print(f"Done applying {DAMethod}")
        print("DA Done")
        return augmentedData, annotationData
    #Helper functions that can be used for testing belows
    def getAugmentedData_Sliced(self, DAMethods, sliceIdx):
        """Augmentate base data in the order the methods is provided. Sliced version
        Args:
            DAMethods ([str]): Array containing DA method names, the order the DA is applied
            sliceIdx (int): Index of slicing
        Returns:
            tuple: 
                - augmentedData [any]: Sliced augmentated training data
                - annotationData [any]: Sliced annotation data
        """
        augmentedData, annotationData = self.getBaseData_Sliced(sliceIdx)
        for DAMethod in DAMethods:
            func = getattr(self._DA, DAMethod)
            if func:
                augmentedData = func(augmentedData)
        return augmentedData, annotationData
    def getBaseData_Sliced(self, sliceIdx):
        """Loads a sliced version of base data
        Args:
            sliceIdx (int): Index of slicing
        Returns:
            tuple:
                - trainingData [any]: Sliced training data
                - annotationData [any]: Sliced annotation data
        """
        trainingData, annotationData = self.getBaseTrainingData()
        if sliceIdx > len(annotationData):
            sliceIdx = len(annotationData)
        return trainingData[0:sliceIdx], annotationData[0:sliceIdx]
    def getValidationData_Sliced(self, sliceIdx):
        """Loads a sliced version of validation data
        Args:
            sliceIdx (int): Index of slicing
        Returns:
            tuple:
                - trainingData [any]: Sliced validation data
                - annotationData [any]: Sliced annotation data
        """
        trainingData, annotationData = self.getValidationData()
        if sliceIdx > len(annotationData):
            sliceIdx = len(annotationData)
        return trainingData[:sliceIdx], annotationData[:sliceIdx]