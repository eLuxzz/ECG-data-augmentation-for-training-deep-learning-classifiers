from dataloader import Dataloader
import argparse
from  dataPlotter import DataPlotter
from dataAugmenter import DataAugmenter

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')


args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv,None,None )
base_data, _ = d.getBaseData_Sliced(1)
augmented_data, _ = d.getAugmentedData_Sliced(["add_powerline_noise", "add_baseline_wander"],1)


plotter = DataPlotter(
    base_data, 
    augmented_data,
    same_graf=False,
    sample_range=[1,5000],
)

plotter.plot()

