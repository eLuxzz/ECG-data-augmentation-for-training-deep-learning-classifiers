from dataloader import Dataloader
import argparse
from  dataPlotter import DataPlotter

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')


args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv,None,None)

base_dataset = d.getBaseData(1)
# augmented_dataset = d.getAugmentedData(["add_baseline_wander"],1)
# augmented_dataset = d.getAugmentedData(["add_gaussian_noise"],1)
augmented_dataset = d.getAugmentedData(["add_powerline_noise"],1)

(b_sample) = base_dataset.take(1)
(a_sample) = augmented_dataset.take(1)

for sample in b_sample:
    b_signal = sample
    print(f"Shape after da: {sample[0].shape}, {sample[1].shape}")    
for sample in a_sample:
    a_signal = sample
    print(f"Shape after da: {sample[0].shape}, {sample[1].shape}")    
f = True
for s in b_signal[0]:
    if  f:
        f = False
        b = s
f = True
for s in a_signal[0]:
    if  f:
        f = False
        a = s
plotter = DataPlotter(
    b,
    a,
    same_graf=True,
    sample_range=[1,5000],
)

plotter.plot()

