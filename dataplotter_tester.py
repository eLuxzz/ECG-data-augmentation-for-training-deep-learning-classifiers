from dataloader import Dataloader
import argparse
from  dataPlotter import DataPlotter

parser = argparse.ArgumentParser(description='Train neural network.')
parser.add_argument('path_train_hdf5', type=str,
                    help='path to hdf5 file containing training tracings')
parser.add_argument('path_train_csv', type=str,
                    help='path to csv file containing training annotations')


args = parser.parse_args()

d = Dataloader(args.path_train_hdf5, args.path_train_csv,None,None, DA_P=5)

base_dataset = d.getTrainingData_Plot(sliceIdx=1)

# augmented_dataset = d.getTrainingData_Plot(["add_gaussian_noise"],1)
# augmented_dataset = d.getTrainingData_Plot(["add_powerline_noise"],1)
augmented_dataset = d.getTrainingData_Plot(["host_guest_augmentation"],5)
# augmented_dataset = d.getTrainingData_Plot(["add_time_warp"],1)

(b_sample) = base_dataset.take(1)
(a_sample) = augmented_dataset.take(1)
# print(b_sample)

# print(a_sample)
for sample in b_sample:
    b_signal = sample
    print(f"Base: Shape: {sample[0].shape}, {sample[1].shape}")    
for sample in a_sample:
    a_signal = sample
    print(f"DA: Shape after da: {sample[0].shape}, {sample[1].shape}")    
f = True
for s in b_signal[0]:
    if f:
        f = False
        b = s
f = True
for d in a_signal[0]:
    if  f:
        f = False
        a = d
# print(b)
# print("-----------------")
# print(a)
plotter = DataPlotter(
    b,
    a,
    same_graf=True,
    #sample_range=[900,1200],
    sample_range=[0,5000]
)
plotter1 = DataPlotter(
    b,
    a,
    same_graf=False,
    sample_range=[0,5000],
)
plotter.plot()
#plotter1.plot()
