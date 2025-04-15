import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        EarlyStopping)
from tensorflow.keras.metrics import AUC
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import time
from CustomCallbacks import UpdateDA
# from CustomMetrics import f1_score_metrics

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
    parser.add_argument("--BS", type=int, default=32,
                        help="Custom Batchsize (Optional)")
    parser.add_argument("--Slice", type=int, default=None,
                        help="Slice dataset (Optional)")
    parser.add_argument("--L2", type=float, default=0,
                        help="L2 value (Optional)")
    parser.add_argument("--ES", type=bool, default=True,
                        help="Early Stopping, default True (Optional)")
    parser.add_argument("--Balanced", type=bool, default=False,
                        help="Use balanced set by over/under sampling")
    args = parser.parse_args()

    # Optimization settings
    lr = 0.002
    batch_size = args.BS
    
    # Load data and setup sequences
    useBalancedSet = args.Balanced
    slice = args.Slice
    buffer_size= 10000
    num_epochs = 50
    dropout_keep_prob = 0.3
    DA_P = 0.85
    DA2_P = 0.25
    l2_lambda = args.L2
    DA_methods = args.DA
    useEarlyStopping = args.ES
    
    dataloader = Dataloader(args.path_train_hdf5, args.path_train_csv,args.path_valid_hdf5, args.path_valid_csv, args.training_dataset_name, 
                            args.validation_dataset_name, batch_size, buffer_size=buffer_size, DA_P=DA_P, useBalancedSet=useBalancedSet)
    validation_dataset = dataloader.getValidationData(sliceIdx=slice)
    if args.DA == []:
        print("Running base set")
    else:
        print(f"Running with Augmentations: {DA_methods}")
    print(f"Batchsize: {batch_size}")
    print(f"L2: {l2_lambda}")
    print(f"Balanced: {useBalancedSet}")
    tf_dataset = dataloader.getTrainingData(DA_methods,sliceIdx=slice)

    loss = 'binary_crossentropy'
    opt = Adam(lr)
    metrics = [AUC(multi_label=True, name ="AUC"),
                AUC(multi_label=True, curve="PR", name="pr_auc")
                # ,f1_score_metrics # Får inte den att funka än, tanken är att byta ut EarlyStopping monitor till F1
                ]
    
    # If you are continuing an interrupted section, uncomment line bellow:
    # PATH_TO_PREV_MODEL = "backup_model_last.keras"
    # model = tf.keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model = get_model(dataloader.n_classes, dropout_keep_prob=dropout_keep_prob, l2_lambda=l2_lambda)
    model.compile(loss=loss, optimizer=opt,  metrics = metrics)

    # Create log
    log_dir = os.path.relpath("logs")  # Välj en enkel sökväg
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    fileName = f"{args.final_model_name}-BS{batch_size}-LR{lr}-DRK_P{dropout_keep_prob}-DA_P{DA_P}-DA2_P{DA2_P}-L2_{l2_lambda}-ES_{useEarlyStopping}-Balanced_{useBalancedSet}-{time_stamp}"
    log_dir = os.path.join(log_dir, f"{fileName}")

    # Callbacks
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   min_lr=1e-6),
                UpdateDA(dataloader)]
    if useEarlyStopping:
        callbacks += [
                    EarlyStopping(monitor="val_loss", # Tanken är att byta ut denna mot F1_Score
                                patience=8,  # Patience should be larger than the one in ReduceLROnPlateau
                                min_delta=0.00001,
                                restore_best_weights = True, 
                                verbose=1),]
    callbacks += [TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)]
        # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.keras'),
                  ModelCheckpoint('./backup_model_best.keras', save_best_only=True),]
    
    # Train neural network
    print("Start training")
    # steps_per_epoch = (slice) // batch_size # Use this when testing with slice.
    # steps_per_epoch = len(trainAnnotationData) // batch_size
    steps_per_epoch = dataloader.datasetLength // batch_size
    # dir_newmodels = os.path.relpath("new_models")
    # os.makedirs(dir_newmodels, exist_ok=True)
    # dir_newmodels = os.path.join(dir_newmodels, f"{fileName}")
    def training():
        # Compute class weights
        # _, trainAnnotationData = dataloader.fileloader.getData(dataloader.PathHDF5_training, dataloader.SetName_Training, dataloader.PathCSV_training)
        # class_weights = compute_class_weight(class_weight="balanced",
        #                                     classes=np.unique(trainAnnotationData.argmax(axis=1)),
        #                                     y=trainAnnotationData.argmax(axis=1))

        # #Convert to dictionary format
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