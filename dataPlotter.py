from dataloader import Dataloader
import argparse
import numpy as np
import matplotlib.pyplot as plt


class DataPlotter:
    def __init__(self, base_data, augmented_data, leads=list(range(12)), sample_range=[0, 5000], same_graf=True):
        """
        base_data: the original ECG data.
        augmented_data: the augmented ECG data.
        leads: List of leads to be plotted, e.g., [0, 1, 5]. If None -> all 12 leads are plotted.
        sample_range: [start, stop] array to select the x-limits of the sample.
        same_graf: Bool, True = base & augmented data plotted on the same subplot, False = separate figures.
        """
        self.base_data = np.squeeze(base_data)[sample_range[0]:sample_range[1]]
        self.augmented_data = np.squeeze(augmented_data)[sample_range[0]:sample_range[1]]
        self.leads = leads
        self.sample_range = sample_range
        self.filtered_base, self.filtered_augmented, self.time = self._filter_data()
        self.same_graf = same_graf

    def _filter_data(self):
        """Filters data based on selected leads and sample range."""
        filtered_base = self.base_data[:, self.leads]
        filtered_augmented = self.augmented_data[:, self.leads]
        time = np.arange(start=self.sample_range[0], stop=self.sample_range[1])
        return filtered_base, filtered_augmented, time

    def _plot_same_figure(self):
        """Plots base & augmented data in the same figure."""
        fig, axes = plt.subplots(len(self.leads), 1, sharex=True)
        
        if len(self.leads) == 1:
            axes = [axes]

        for idx, lead in enumerate(self.leads):
            axes[idx].plot(self.time, self.filtered_base[:, idx], label="Base Data", color="black")
            axes[idx].plot(self.time, self.filtered_augmented[:, idx], label="Augmented Data", linestyle="dashed", color="red")
            axes[idx].set_title(f"Lead {lead + 1}")
            axes[idx].legend(loc='upper right')
            axes[idx].set_ylabel("mV")

        axes[-1].set_xlabel("Samples")
        plt.suptitle("ECG Data (Base vs. Augmented)")
        plt.tight_layout()
        plt.show()

    def _plot_separate_figures(self):
        """Plots base and augmented data in separate figures."""
        fig1, axes1 = plt.subplots(len(self.leads), 1, sharex=True)
        fig2, axes2 = plt.subplots(len(self.leads), 1, sharex=True)

        if len(self.leads) == 1:
            axes1 = [axes1]
            axes2 = [axes2]

        for idx, lead in enumerate(self.leads):
            axes1[idx].plot(self.time, self.filtered_base[:, idx], color="black")
            axes1[idx].set_title(f"Base Data - Lead {lead + 1}")
            axes1[idx].set_ylabel("mV")
            
            axes2[idx].plot(self.time, self.filtered_augmented[:, idx], color="red")
            axes2[idx].set_title(f"Augmented Data - Lead {lead + 1}")
            axes2[idx].set_ylabel("mV")

        axes1[-1].set_xlabel("Samples")
        axes2[-1].set_xlabel("Samples")

        fig1.suptitle("Base ECG Data")
        fig2.suptitle("Augmented ECG Data")

        plt.tight_layout()
        plt.show()

    def plot(self):
        """Determines which plotting method to use."""
        if self.same_graf:
            self._plot_same_figure()
        else:
            self._plot_separate_figures()
