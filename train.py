from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        EarlyStopping)
import argparse
import os
from model import get_model
from datasets import ECGSequence
from dataloader import Dataloader

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
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--training_dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--validation_dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()

    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.01
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
    # Load data and setup sequences
    slice = 200
    dataloader = Dataloader(args.path_train_hdf5, args.path_train_csv,args.path_valid_hdf5, args.path_valid_csv, args.training_dataset_name, args.validation_dataset_name)
    trainSignalData, trainAnnotationData = dataloader.getAugmentedData_Sliced(["add_baseline_wander", "add_powerline_noise", "add_gaussian_noise"], slice)
    validationSignalData, validationSignalAnnotations = dataloader.getValidationData_Sliced(slice)

    train_seq = ECGSequence(trainSignalData, trainAnnotationData, batch_size)
    valid_seq = ECGSequence(validationSignalData, validationSignalAnnotations, batch_size)

    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(train_seq.n_classes)
    model.compile(loss=loss, optimizer=opt)
    # Create log

    log_dir = os.path.relpath("logs")  # Välj en enkel sökväg
    os.makedirs(log_dir, exist_ok=True)

    callbacks += [TensorBoard(log_dir=log_dir, write_graph=False)]
        # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.keras'),
                  ModelCheckpoint('./backup_model_best.keras', save_best_only=True)]
    # Train neural network
    history = model.fit(train_seq,
                        epochs=20,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)
    # Save final result
    model.save("./final_model.keras")
