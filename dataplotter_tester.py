from dataloader import Dataloader
from  dataPlotter import DataPlotter
import argparse

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')
args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv, None,None, DA_P=1)

base_dataset = d.getTrainingData_Plot(sliceIdx=10)
augmented_dataset = d.getTrainingData_Plot(["amplitude_scaling"],10)

(b_sample) = base_dataset.take(1)
(a_sample) = augmented_dataset.take(1)
for sample in b_sample:
    b_signal = sample
for sample in a_sample:
    a_signal = sample

base = b_signal[0][0]
aug = a_signal[0][0]

plotter = DataPlotter(
    base,
    aug,
    same_graf=True,
    # leads=list([2]), #% Use this if you want to plot specific leads
    sample_range=[0,5000],
)
plotter.plot()

