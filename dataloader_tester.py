from dataloader import Dataloader
import argparse

def main():
     # Get data and train
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
    
    dataloader = Dataloader(args.path_train_hdf5, args.path_train_csv,args.path_valid_hdf5, args.path_valid_csv, args.training_dataset_name, 
                            args.validation_dataset_name)

    dataset = dataloader.getTrainingData(["time_warp_crop"], sliceIdx=60)

    print(dataset)
if __name__ == "__main__":    
    main()