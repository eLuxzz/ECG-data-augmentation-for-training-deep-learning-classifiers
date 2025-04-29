from fileloader import Fileloader
from dataAugmenter import DataAugmenter
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import gc
class Dataloader():
    def __init__(self, pathHDF5_training, pathCSV_training, pathHDF5_valid, pathCSV_valid, 
                 setNameTraining = "tracings", setNameValid="tracings",
                 batch_size=32, buffer_size=10000, 
                 DA_P = 0.8,
                 DA2_P = 0.25,
                 useBalancedSet = False):
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
        self.UseBalancedSet = useBalancedSet
        self.n_classes = None
        self._DA = DataAugmenter()
        self.fileloader = Fileloader()
        self.DA_Methods = []

        self.DA_index = tf.Variable(0, trainable=False, dtype=tf.int32)

    def __getBaseDataset(self, sliceIdx=None):
        """Loads and returns a base dataset
        Returns:
            tuple:
                - signalData_training [any]: Base training data
                - annotationData_training [any]: Annotation data
        """
        signal_training, labels_training = self.fileloader.getData(self.PathHDF5_training, self.SetName_Training, self.PathCSV_training)
        
        self.n_classes = labels_training.shape[1]
        
        balanced_idx = self.get_oversampled_indices(labels_training) if self.UseBalancedSet is True else None
        # balanced_idx = self.get_balanced_indices(labels_training) if self.UseBalancedSet is True else None
        
        n_balancedIdx = self.datasetLength = len(balanced_idx if self.UseBalancedSet is True else labels_training)
        if self.UseBalancedSet: # Prints for dataset distribution
            print(len(labels_training),len(balanced_idx))
            org_samples = []
            for k in range(5):
                org_samples.append(len(np.where(labels_training[:, k] > 0 )[0]))
            print(f"Original: {org_samples}")
            new_samples = []
            for i in range(5):
                c = 0
                for k in balanced_idx:
                    if labels_training[k, i] > 0:
                        c+=1
                new_samples.append(c)
            print(f"Total: {new_samples}")

        chunk_size = n_balancedIdx // 5
        print(n_balancedIdx)
        def dataset_generator():
            if sliceIdx:
                for i in range(n_balancedIdx if sliceIdx is None else sliceIdx):
                    yield signal_training[i], labels_training[i]
            else:
                if not self.UseBalancedSet:
                    print("Using default dataset")
                    i = 0
                    while True:
                        yield signal_training[i], labels_training[i]
                        i += 1
                        if i >= n_balancedIdx:
                            i = 0 
                else:
                    print("Using balanced dataset")
                    i = 0
                    end_idx = min(i + chunk_size, n_balancedIdx)
                    while True:
                        idx = balanced_idx[i]
                        yield signal_training[idx], labels_training[idx]
                        i += 1
                        if i >= n_balancedIdx:
                            i = 0 
                        
              
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
            if self.DAMethods_len == 0:
                return
            index = tf.random.uniform((), 0, self.DAMethods_len, dtype=tf.int32)
            self.DA_index.assign(index) # tf.variable so updates are seen when running in graph mode w map() func
            # print(f"\n Selected DA method: {self.DA_Methods[index]}")
    
    def get_balanced_indices(self, labels, confidence_threshold=0.8, other_threshold=0.35):
        y_classes = np.argmax(labels, axis=1)

        # ---- 1. Downsample NORM (class 0) ----
        norm_indices = np.where(y_classes == 0)[0]
        np.random.shuffle(norm_indices)
        norm_indices = norm_indices[:5000]

        # ---- 2. Confident HYP Samples (class 2) ----
        hyp_mask = (labels[:, 2] >= confidence_threshold) & \
                (labels[:, [0,1,3,4]] <= other_threshold).all(axis=1)
        confident_hyp_indices = np.where(hyp_mask)[0]
        print(f"After filter: {len(confident_hyp_indices)}")
        # If confident HYP is fewer than target, oversample from them
        target_hyp = 1500
        current_hyp = len(confident_hyp_indices)

        if current_hyp < target_hyp:
            ros = RandomOverSampler(sampling_strategy={2: target_hyp})
            dummy_indices = norm_indices[:1]
            dummy_labels = np.full(len(dummy_indices), 0)
            
            hyp_labels = np.full(len(confident_hyp_indices), 2)
            
            all_indices = np.concatenate([confident_hyp_indices, dummy_indices])
            all_labels = np.concatenate([hyp_labels, dummy_labels])
            oversampled_hyp_indices, oversampled_labels = ros.fit_resample(all_indices.reshape(-1, 1),
                                                        all_labels)
            hyp_indices_final = oversampled_hyp_indices[oversampled_labels == 2].flatten()
        else:
            np.random.shuffle(confident_hyp_indices)
            hyp_indices_final = confident_hyp_indices[:target_hyp]
        
        # ---- 3. Include all other classes as-is ----
        cd_indices  = np.where(y_classes == 1)[0]
        hyp_indices  = np.where(y_classes == 2)[0]
        mi_indices  = np.where(y_classes == 3)[0]
        sttc_indices = np.where(y_classes == 4)[0]

        final_indices = np.concatenate([
            np.arange(len(labels)),
            hyp_indices_final
            # norm_indices,
            # cd_indices,
            # mi_indices,
            # sttc_indices,
            # hyp_indices,
            # hyp_indices_final
        ])
        np.random.shuffle(final_indices)
        return final_indices
    def get_oversampled_indices(self, labels):
        """
        Uses RandomOverSampler to generate balanced indices.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            np.ndarray: Oversampled indices for balanced dataset.
        """
        TARGET_HYP_SAMPLES = 3000
        TARGET_CD_SAMPLES = 4500
        y_classes = np.argmax(labels, axis=1)  # Convert one-hot labels to class indices
        indices = np.arange(len(labels))  # Original indices

        # Count existing HYP samples
        hyp_indices = np.where(y_classes == 2)[0]
        current_hyp_count = len(hyp_indices)

        if current_hyp_count >= TARGET_HYP_SAMPLES:
            return indices  # No need for oversampling

        # Define custom sampling strategy (only oversample HYP)
        sampling_strategy = {cls: count for cls, count in zip(*np.unique(y_classes, return_counts=True))}
        sampling_strategy[1] = TARGET_CD_SAMPLES  # Set CD target
        sampling_strategy[2] = TARGET_HYP_SAMPLES  # Set HYP target

        ros = RandomOverSampler(sampling_strategy=sampling_strategy)
        oversampled_indices, _ = ros.fit_resample(indices.reshape(-1, 1), y_classes)

        return oversampled_indices.flatten()
    
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
        self.DAMethods_len = len_DAMethods
        def apply_augmentation(x, y):  # Tensorflow runs this in graph mode and only trace it once. Any changes in variables won't be seen, unless it is an tf.Variable
            if len_DAMethods == 0:
                return x,y
            # if y[2] > 0 or tf.random.uniform((), seed=42, dtype=tf.float16) < self.DA_P:  # Randomly apply augmentations
            if tf.random.uniform((), seed=42, dtype=tf.float16) < self.DA_P:  # Randomly apply augmentations
                index = self.DA_index.read_value()
                for i in range(0, len_DAMethods):
                    if index != i:
                        continue
                    func = getattr(self._DA, DAMethods[i])
                    x,y = func(x,y)

                    if tf.random.uniform((), seed=42, dtype=tf.float16) < self.DA2_P:
                        index2 = tf.random.uniform((), 0, len_DAMethods, seed=42, dtype=tf.int32)
                        for j in range(0, len_DAMethods):
                            if index2 == j and j!=i:
                                func2 = getattr(self._DA, DAMethods[j])
                                x,y = func2(x,y)
            return x, y
        
        # # First Epoch: No DA, just shuffled & batched
        # base_epoch = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
        #                 .repeat(1) \
        #                 .batch(self.batch_size) \
        #                 .prefetch(tf.data.AUTOTUNE) \

        # Followihng Epochs: DA   
        dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True, seed=42) \
            .repeat() \
            .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        
        print("Base epoch loaded and DA Compiled")
        # for sample in dataset.take(1):
        #     print(f"DA: Shape after da: {sample[0].shape}, {sample[1].shape}")   
        # final_dataset = base_epoch.concatenate(dataset)
        return dataset
    
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
                if func:  # Randomly apply augmentations
                    x, y = func(x, y)
            return x, y
        

        # Following Epochs: DA   
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
        signalData_validation, annotationData_validation = self.fileloader.getData(self.PathHDF5_valid,self.SetName_Valid, self.PathCSV_valid)
        if sliceIdx:
            signalData_validation = signalData_validation[:sliceIdx]
            annotationData_validation = annotationData_validation[:sliceIdx]

        dataset = tf.data.Dataset.from_tensor_slices((signalData_validation, annotationData_validation))
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        print("Validation data loaded")
        return dataset
        