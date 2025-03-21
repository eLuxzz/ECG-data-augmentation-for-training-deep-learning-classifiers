from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        EarlyStopping)
from tensorflow.keras import mixed_precision
from tensorflow import keras

import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import os
import logging
import time

from model import get_model
from datasets import ECGSequence
from dataloader import Dataloader
from fileloader import Fileloader

logging.getLogger("tensorflow").setLevel(logging.ERROR)
mixed_precision.set_global_policy('mixed_float16')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
    args = parser.parse_args()

    # Optimization settings
    lr = 0.001
    batch_size = 32
    
    # Load data and setup sequences
    slice = None
    buffer_size= 10000
    num_epochs = 40
    dropout_keep_prob = 0.7
    fileloader = Fileloader()

    trainSignalData, trainAnnotationData = fileloader.getData(args.path_train_hdf5,"tracings", args.path_train_csv)
    train_seq = ECGSequence(trainSignalData, trainAnnotationData, batch_size)

    dataloader = Dataloader(args.path_train_hdf5, args.path_train_csv,args.path_valid_hdf5, args.path_valid_csv, args.training_dataset_name, 
                            args.validation_dataset_name, batch_size, buffer_size=buffer_size,epochs=num_epochs)
    # tf_dataset = dataloader.getBaseData(sliceIdx=slice)
    valid_seq = dataloader.getValidationData(sliceIdx=slice)
    # tf_dataset = dataloader.getAugmentedData(["add_powerline_noise"], sliceIdx=slice)
    tf_dataset = dataloader.getAugmentedData(["add_baseline_wander", "add_powerline_noise", "add_gaussian_noise"],sliceIdx=slice)    

    # If you are continuing an interrupted section, uncomment line bellow:
    # PATH_TO_PREV_MODEL = "backup_model_last.keras"
    # model = tf.keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    loss = 'binary_crossentropy'
    opt = Adam(lr)
    model = get_model(train_seq.n_classes, dropout_keep_prob=dropout_keep_prob)
    model.compile(loss=loss, optimizer=opt)

    # Create log
    log_dir = os.path.relpath("logs")  # Välj en enkel sökväg
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"{args.final_model_name}-BS{batch_size}-LR{lr}-{time.strftime("%Y%m%d-%H%M%S")}")
    # tf.keras.backend.clear_session()
    # tf.profiler.experimental.start(log_dir)
    # tensorflow.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

    # Callbacks
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   min_lr=lr / 1000),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
    callbacks += [TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)]
        # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.keras'),
                  ModelCheckpoint('./backup_model_best.keras', save_best_only=True)]
    
    # Train neural network
    print("Start training")
    # steps_per_epoch = (slice) // batch_size # Use this when testing with slice.
    steps_per_epoch = len(trainAnnotationData) // batch_size
    
    def training_no_sequence():
        history = model.fit(tf_dataset,
                            epochs=num_epochs,
                            initial_epoch=0,  # If you are continuing a interrupted section change here
                            callbacks=callbacks,
                            validation_data=valid_seq,
                            verbose=1,
                            steps_per_epoch=steps_per_epoch)
    # Save final result
    training_no_sequence()
    model.save(f"./{args.final_model_name}-BS{batch_size}-LR{lr}-DR{dropout_keep_prob}-{time.strftime("%Y%m%d-%H%M%S")}.keras")
    K.clear_session()
    # tf.profiler.experimental.stop()