from dataloader import Dataloader
import argparse
from  dataPlotter import DataPlotter

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')


args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv,None,None )
base_data, _ = d.getBaseData_Sliced(1)
augmented_data = base_data + 1 

print(len(base_data))



plotter = DataPlotter(
    base_data, 
    augmented_data,
    same_graf=True,
    sample_range=[1,500],
    leads=[1,2]

)

plotter.plot()

