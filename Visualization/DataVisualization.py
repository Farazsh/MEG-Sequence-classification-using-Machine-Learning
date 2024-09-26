# Imports
import mne
import numpy as np
import matplotlib.pyplot as plt
import warnings
from os.path import join

warnings.filterwarnings('ignore')


class MEGAnalysis:
    def __init__(self, data_dir, filename):
        self.data_dir = data_dir
        self.filename = filename
        self.raw_file = join(self.data_dir, self.filename)
        self.epochs = None
        self.events = None

    def load_data(self):
        """Loads the MEG epochs data from the file."""
        self.epochs = mne.read_epochs(self.raw_file)
        self.events = self.epochs.events

    def plot_events(self):
        """Plots the events."""
        if self.events is not None:
            mne.viz.plot_events(self.events)
        else:
            print("Events data is not loaded.")

    def plot_average(self):
        """Plots the average of all epochs."""
        if self.epochs is not None:
            self.epochs.average().plot()
        else:
            print("Epochs data is not loaded.")

    def print_event_statistics(self):
        """Prints the percentage of different events."""
        if self.epochs is not None:
            event_labels = [
                'cry_real_10', 'cry_omission_10', 'bird_real_10',
                'phone_real_10', 'phone_omission_10', 'bell_real_10'
            ]

            for label in event_labels:
                percentage = np.around(len(self.epochs[label]) / len(self.epochs), decimals=2)
                print(f"Percentage of {label} events : {percentage}")
        else:
            print("Epochs data is not loaded.")

    def plot_condition_averages(self, channel=18):
        """Plots the average waveforms for different conditions using Matplotlib."""
        if self.epochs is not None:
            conditions = {
                'cry_real_10': 'blue',
                'cry_omission_10': 'red',
                'bird_real_10': 'green',
                'phone_real_10': 'violet',
                'phone_omission_10': 'yellow',
                'bell_real_10': 'pink'
            }

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel('Time Instances')
            ax.set_ylabel('Volt')

            for condition, color in conditions.items():
                data = self.epochs[condition].average().data[channel, :]
                ax.plot(data, color=color, label=condition)

            legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
            plt.title('ERP of different conditions')
            plt.show()
        else:
            print("Epochs data is not loaded.")

    def plot_topomap(self, times=np.arange(-0.1, 0.5, 0.1), ch_type='mag'):
        """Plots the topomap at different times."""
        if self.epochs is not None:
            self.epochs.average().plot_topomap(times, ch_type=ch_type)
        else:
            print("Epochs data is not loaded.")


def main():
    data_dir = r"\ML on MEG\SensoryProcessing_MEG-faraz\MEG_Data\S17"
    filename = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR1.fif"

    meg_analysis = MEGAnalysis(data_dir, filename)
    meg_analysis.load_data()
    meg_analysis.plot_events()
    meg_analysis.plot_average()
    meg_analysis.print_event_statistics()
    meg_analysis.plot_condition_averages()
    meg_analysis.plot_topomap()


if '__name__==__main__':
    main()
