from fileloader import Fileloader
from dataAugmenter import DataAugmenter
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
class Dataloader():
    def __init__(self, pathHDF5_training, pathCSV_training, pathHDF5_valid, pathCSV_valid, 
                 setNameTraining = "tracings", setNameValid="tracings",
                 batch_size=32, buffer_size=10000, 
                 DA_P = 0.8,
                 DA2_P = 0.25):
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
        self.current_epoch = 0
        self.datasetLength = 0
        self.DA_P = tf.cast(DA_P, dtype=tf.float16)
        self.DA2_P = tf.cast(DA2_P, dtype=tf.float16)

        self.n_classes = None
        self._DA = DataAugmenter()
        self._Fileloader = Fileloader()
        self.DA_Methods = []

        self.DA_index = tf.Variable(0, trainable=False, dtype=tf.int32)

    def __getBaseDataset(self, sliceIdx=None):
        """Loads and returns a base dataset
        Returns:
            tuple:
                - signalData_training [any]: Base training data
                - annotationData_training [any]: Annotation data
        """
        signalData_training, annotationData_training = self._Fileloader.getData(self.PathHDF5_training, self.SetName_Training, self.PathCSV_training)
        
        self.datasetLength = len(annotationData_training)
        self.n_classes = annotationData_training.shape[1]
        
        def dataset_generator():
            for i in range(len(signalData_training) if sliceIdx is None else sliceIdx):
                # yield tf.cast(signalData_training[i], dtype=tf.float16), tf.cast(annotationData_training[i], dtype=tf.float16)
                yield signalData_training[i], annotationData_training[i]

        # Wrap the generator in tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                # tf.TensorSpec(shape=(5000, 12), dtype=tf.float16),
                # tf.TensorSpec(shape=(5), dtype=tf.float16)
                tf.TensorSpec(shape=(5000, 12), dtype=tf.float32),
                tf.TensorSpec(shape=(5), dtype=tf.float32)
                )
        )
        return dataset
    def update_DAMethod_index(self):
            index = tf.random.uniform((), 0, len(self.DA_Methods), dtype=tf.int32)
            self.DA_index.assign(index) # tf.variable so updates are seen when running in graph mode w map() func
            print(f"\n Selected DA method: {self.DA_Methods[index]}")

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
        self.DA_Methods = DAMethods
        len_DAMethods = len(DAMethods)
        def apply_augmentation(x, y):  # Tensorflow runs this in graph mode and only trace it once. Any changes in variables won't be seen, unless it is an tf.Variable
            if tf.random.uniform((), dtype=tf.float16) < self.DA_P:  # Randomly apply augmentations
                index = self.DA_index.read_value()  # Correct way to read tf.Variable in graph mode
                for i in range(0, len_DAMethods):
                    if index != i:
                        continue
                    func = getattr(self._DA, DAMethods[i])
                    x,y = func(x,y)

                    if tf.random.uniform((), dtype=tf.float16) < self.DA2_P:
                        index2 = tf.random.uniform((), 0, len_DAMethods, dtype=tf.int32)
                        for j in range(0, len_DAMethods):
                            if index2 == j and j!=i:
                                func2 = getattr(self._DA, DAMethods[j])
                                x,y = func2(x,y)
            return x, y
            
            
        
        # First Epoch: No DA, just shuffled & batched
        base_epoch = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
                        .repeat(5) \
                        .batch(self.batch_size) \
                        .prefetch(tf.data.AUTOTUNE) \

        # Followihng Epochs: DA   
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
            .repeat() \
            .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        
        print("Base epoch loaded and DA Compiled")
        # for sample in dataset.take(1):
        #     print(f"DA: Shape after da: {sample[0].shape}, {sample[1].shape}")   
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
                if func and tf.random.uniform((), dtype=tf.float16) < self.DA_P:  # Randomly apply augmentations
                    x, y = func(x, y)
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
        if sliceIdx:
            signalData_validation = signalData_validation[:sliceIdx]
            annotationData_validation = annotationData_validation[:sliceIdx]

        dataset = tf.data.Dataset.from_tensor_slices((signalData_validation, annotationData_validation))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print("Validation data loaded")
        return dataset
        