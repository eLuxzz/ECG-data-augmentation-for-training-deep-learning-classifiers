import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        EarlyStopping)
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import numpy as np
import time
from CustomCallbacks import UpdateDA

from model import get_model
from dataloader import Dataloader


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_train_hdf5', type=str,
                        help='path to hdf5 file containing training tracings')
    parser.add_argument('path_train_csv', type=str,
                        help='path to csv file containing training annotations')
    parser.add_argument('path_valid_hdf5', type=str,
                        help='path to hdf5 file containing validation tracings')
    parser.add_argument('path_valid_csv', type=str,
                        help='path to csv file containing validation annotations')
    parser.add_argument('--training_dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--validation_dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--final_model_name', type=str, default='final_model',
                        help='name of the output file')
    parser.add_argument("--DA", type=str, nargs="+", default=[],
                        help="name of DA methods to apply, leave empty for base set")

    args = parser.parse_args()

    # Optimization settings
    lr = 0.002
    batch_size = 32
    
    # Load data and setup sequences
    slice = None
    buffer_size= 10000
    num_epochs = 70
    dropout_keep_prob = 0.5
    DA_P = 0.85
    DA2_P = 0.25
    l2_lambda = 0
    DA_methods = args.DA
    tf.keras.callbacks.Callback()
    dataloader = Dataloader(args.path_train_hdf5, args.path_train_csv,args.path_valid_hdf5, args.path_valid_csv, args.training_dataset_name, 
                            args.validation_dataset_name, batch_size, buffer_size=buffer_size, DA_P=DA_P)
    validation_dataset = dataloader.getValidationData(sliceIdx=slice)
    if args.DA == []:
        print("Running base set")
    else:
        print(f"Running with Augmentations: {DA_methods}")
    tf_dataset = dataloader.getTrainingData(DA_methods,sliceIdx=slice)
    dataloader.update_DAMethod_index()
    

    loss = 'binary_crossentropy'
    opt = Adam(lr)

    # If you are continuing an interrupted section, uncomment line bellow:
    # PATH_TO_PREV_MODEL = "backup_model_last.keras"
    # model = tf.keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(dataloader.n_classes, dropout_keep_prob=dropout_keep_prob, l2_lambda=l2_lambda)
    model.compile(loss=loss, optimizer=opt,  metrics = ["accuracy"])

    # Create log
    log_dir = os.path.relpath("logs")  # Välj en enkel sökväg
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    fileName = f"{args.final_model_name}-BS{batch_size}-LR{lr}-DR_Keep_P{dropout_keep_prob}-DA_P{DA_P}-DA2_P{DA2_P}-L2_{l2_lambda}-{time_stamp}"
    log_dir = os.path.join(log_dir, f"{fileName}")

    # Callbacks
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=3,
                                   min_lr=lr / 10000),
                 EarlyStopping(patience=8,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001,
                               restore_best_weights = True, 
                               verbose=1),
                UpdateDA(dataloader)]
    callbacks += [TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, profile_batch=10)]
        # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.keras'),
                  ModelCheckpoint('./backup_model_best.keras', save_best_only=True),]
    
    # Train neural network
    print("Start training")
    # steps_per_epoch = (slice) // batch_size # Use this when testing with slice.
    # steps_per_epoch = len(trainAnnotationData) // batch_size
    steps_per_epoch = dataloader.datasetLength // batch_size
    
    def training():
        # # Compute class weights
        # class_weights = compute_class_weight(class_weight="balanced",
        #                                     classes=np.unique(trainAnnotationData.argmax(axis=1)),
        #                                     y=trainAnnotationData.argmax(axis=1))

        # Convert to dictionary format
        # class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        # print(class_weights_dict)
        
        history = model.fit(tf_dataset,
                            epochs=num_epochs,
                            # class_weight = class_weights_dict,
                            initial_epoch=0,  # If you are continuing a interrupted section change here
                            callbacks=callbacks,
                            validation_data=validation_dataset,
                            verbose=1,
                            steps_per_epoch=steps_per_epoch,
                           )
        model.save(f"./{fileName}.keras")
    # Save final result
    training()
    K.clear_session()
    print(f"Training finished, saved as {fileName}")
    # tf.profiler.experimental.stop()