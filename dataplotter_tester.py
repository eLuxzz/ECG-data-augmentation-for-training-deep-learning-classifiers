from dataloader import Dataloader
import argparse
from  dataPlotter import DataPlotter
from dataAugmenter import DataAugmenter
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')


args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv,None,None)

base_dataset = d.getBaseData_Sliced(1)
augmented_dataset = d.getAugmentedData_Sliced(["add_powerline_noise", "add_baseline_wander"],1)

(b_sample) = base_dataset.take(1)
(a_sample) = augmented_dataset.take(1)
print(augmented_dataset)
print(b_sample)

# for image, label in b_sample:
#     print(image, label)
# plotter = DataPlotter(
#     b_sample,
#     a_sample,
#     same_graf=True,
#     sample_range=[1,5000],
# )

# plotter.plot()

