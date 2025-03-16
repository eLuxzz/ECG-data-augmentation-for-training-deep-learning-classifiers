from fileloader import Fileloader
from dataAugmenter import DataAugmenter
from datasets import ECGSequence
import pdb
import tensorflow as tf
class Dataloader():
    def __init__(self, pathHDF5_training, pathCSV_training, pathHDF5_valid, pathCSV_valid, setNameTraining = "tracings", setNameValid="tracings",batch_size=32):
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
        self.batch_size = batch_size
        self._DA = DataAugmenter()
        self._Fileloader = Fileloader()
        
    def getBaseData(self, sliceIdx=None):
        """Loads and returns data without augmentation
        Returns:
            tuple:
                - signalData_training [any]: Base training data
                - annotationData_training [any]: Annotation data
        """
        signalData_training, annotationData_training = self._Fileloader.getData(self.PathHDF5_training, self.SetName_Training, self.PathCSV_training)
        train_seq = ECGSequence(signalData_training[0:sliceIdx], annotationData_training[0:sliceIdx], batch_size=self.batch_size)
        def dataset_generator():
            for i in range(len(train_seq)):
                yield train_seq[i]

        # Wrap the generator in tf.data.Dataset
        tf_dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(None,5000, 12), dtype=tf.float16),
                tf.TensorSpec(shape=(None,5), dtype=tf.float16)
                )
        )

        return tf_dataset
    def getValidationData(self, sliceIdx = None):
        """Loads and return validation data

        Returns:
            signalData_validation: Signal Data for validation
            annoatationData_validation: Annotation Data
        """
        print("Getting validation data")
        signalData_validation, annotationData_validation = self._Fileloader.getData(self.PathHDF5_valid,self.SetName_Valid, self.PathCSV_valid)
        print("Validation data loaded")
        valid_seq = ECGSequence(signalData_validation[0:sliceIdx], annotationData_validation[0:sliceIdx], self.batch_size)
        return valid_seq
    def getAugmentedData(self, DAMethods, sliceIdx = None):
        """Augmentate base data in the order the methods is provided.
        Args:
            DAMethods ([str]): Array containing DA method names, the order the DA is applied
        Returns:
            tuple:
                - augmentedData [any]: Augmentated training data
                - annotationData [any] : Annotation data
        """
        base_dataset = self.getBaseData(sliceIdx)
        print("Base-dataset loaded")
        new_dataset = base_dataset
        for DAMethod in DAMethods:
            func = getattr(self._DA, DAMethod)
            if func:
                print(f"Applying {DAMethod}")
                da_dataset = base_dataset.map(func, num_parallel_calls=tf.data.AUTOTUNE)
                new_dataset = new_dataset.concatenate(da_dataset)

                for sample in da_dataset.take(1):
                    print(f"Shape after {DAMethod}: {sample[0].shape}, {sample[1].shape}")
                print(f"Done applying {DAMethod}")
        print("DA Done")
        buffer_size = 1000
        # new_dataset = new_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        return new_dataset
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
        return self.getAugmentedData(DAMethods, sliceIdx)
    
    def getBaseData_Sliced(self, sliceIdx):
        """Loads a sliced version of base data
        Args:
            sliceIdx (int): Index of slicing
        Returns:
            tuple:
                - trainingData [any]: Sliced training data
                - annotationData [any]: Sliced annotation data
        """
        return self.getBaseData(sliceIdx)
    def getValidationData_Sliced(self, sliceIdx):
        """Loads a sliced version of validation data
        Args:
            sliceIdx (int): Index of slicing
        Returns:
            tuple:
                - trainingData [any]: Sliced validation data
                - annotationData [any]: Sliced annotation data
        """
        
        return self.getValidationData(sliceIdx)