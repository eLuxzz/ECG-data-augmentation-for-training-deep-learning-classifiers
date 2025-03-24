from fileloader import Fileloader
from dataAugmenter import DataAugmenter
import tensorflow as tf

class Dataloader():
    def __init__(self, pathHDF5_training, pathCSV_training, pathHDF5_valid, pathCSV_valid, 
                 setNameTraining = "tracings", setNameValid="tracings",
                 batch_size=32, buffer_size=10000, epochs = 20, DA_P = 0.4):
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
        self.buffer_size = buffer_size
        self.epochs = epochs
        self.DA_P = DA_P
        self._DA = DataAugmenter()
        self._Fileloader = Fileloader()

    def __getBaseDataset(self, sliceIdx=None):
        """Loads and returns a base dataset
        Returns:
            tuple:
                - signalData_training [any]: Base training data
                - annotationData_training [any]: Annotation data
        """
        signalData_training, annotationData_training = self._Fileloader.getData(self.PathHDF5_training, self.SetName_Training, self.PathCSV_training)
        signalData_training = signalData_training[:sliceIdx]
        annotationData_training = annotationData_training[:sliceIdx]
        def dataset_generator():
            for i in range(len(signalData_training) if sliceIdx is None else sliceIdx):
                yield signalData_training[i], annotationData_training[i]

        # Wrap the generator in tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(5000, 12), dtype=tf.float32),
                tf.TensorSpec(shape=(5), dtype=tf.float32)
                )
        )
        return dataset
    
    def getTrainingData(self, DAMethods = [], sliceIdx = None):
        """Augmentate base data in the order the methods is provided.
        Args:
            DAMethods ([str]): Array containing DA method names, the order the DA is applied
        Returns:
            tuple:
                - augmentedData [any]: Augmentated training data
                - annotationData [any] : Annotation data
        """
        dataset = self.__getBaseDataset(sliceIdx)

        def apply_augmentation(x, y):
            for aug in DAMethods:
                func = getattr(self._DA, aug)
                if func and tf.random.uniform(()) < self.DA_P:  # Randomly apply augmentations
                    x, y = func(x, y)
            return x, y
        
        # First Epoch: No DA, just shuffled & batched
        base_epoch = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
                        .repeat(1) \
                        .batch(self.batch_size) \
                        .prefetch(tf.data.AUTOTUNE) \

        # Followihng Epochs: DA   
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
            .repeat() \
            .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        
        print("Base epoch loaded and DA Compiled")
        for sample in dataset.take(1):
            print(f"DA: Shape after da: {sample[0].shape}, {sample[1].shape}")   
        final_dataset = base_epoch.concatenate(dataset)
        return final_dataset
    
    def getTrainingData_Plot(self, DAMethods = [], sliceIdx = None):
        """Augmentate base data in the order the methods is provided.
        Args:
            DAMethods ([str]): Array containing DA method names, the order the DA is applied
        Returns:
            tuple:
                - augmentedData [any]: Augmentated training data
                - annotationData [any] : Annotation data
        """
        dataset = self.__getBaseDataset(sliceIdx)

        def apply_augmentation(x, y):
            for aug in DAMethods:
                func = getattr(self._DA, aug)
                if func and tf.random.uniform(()) < self.DA_P:  # Randomly apply augmentations
                    
                    x, y = func(x, y)
                tf.print(x)
            return x, y
        

        # Followihng Epochs: DA   
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE) 

        return dataset
    def getValidationData(self, sliceIdx = None):
        """Loads and return validation data

        Returns:
            signalData_validation: Signal Data for validation
            annoatationData_validation: Annotation Data
        """
        signalData_validation, annotationData_validation = self._Fileloader.getData(self.PathHDF5_valid,self.SetName_Valid, self.PathCSV_valid)
        dataset = tf.data.Dataset.from_tensor_slices((signalData_validation[0:sliceIdx], annotationData_validation[0:sliceIdx]))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        print("Validation data loaded")
        return dataset
        