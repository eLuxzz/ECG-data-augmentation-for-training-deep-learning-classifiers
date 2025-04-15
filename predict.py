import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence
from fileloader import Fileloader

def makePrediction(modelPath, testSignalHdf5, datasetName):
     # Import data
    fileloader = Fileloader()
    signaldata, _ = fileloader.getData(testSignalHdf5, datasetName)
    seq = ECGSequence(signaldata, batch_size=32)
    
    # Import model
    model = load_model(modelPath, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)
    return y_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str, #signals_test.hdf5
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model', #model.keras 
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_folder', default="dnn_predicts",
                        help='Output folder.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    y_score = makePrediction(args.path_to_model, args.path_to_hdf5, args.dataset_name)
    
    fileName = args.path_to_model.split("/")[-1]
    output_path = os.path.join(args.output_folder, f"{str.removesuffix(fileName, ".keras")}")

    np.save(output_path, y_score)

    print("Output predictions saved")
