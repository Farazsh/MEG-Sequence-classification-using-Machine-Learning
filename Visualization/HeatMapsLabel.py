import os
import warnings

import numpy as np
import mne
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')


def load_epochs_data(data_dir, filename):
    """Load epochs data from a .npy file."""
    epochs_file = os.path.join(data_dir, filename)
    epochs = np.load(epochs_file, allow_pickle=True)
    return epochs


def get_model_data(epochs, model_number):
    """Extract and transpose data for the specified model."""
    return epochs[model_number, :, :].transpose()


def create_model_plot(model_data, model_number, save_fig=False, save_path=None):
    """Create a plot of the model data."""
    fig, ax = plt.subplots()
    im = ax.matshow(model_data, origin='lower')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Timepoints (secs)')
    ax.set_ylabel('Epochs')
    ax.set_title(f'Performance of Model {model_number} over '
                 f'{model_data.shape[0]} timepoints for {model_data.shape[1]} epochs')
    plt.colorbar(im, ax=ax)
    fig.set_size_inches(8, 25)
    if save_fig and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()


def main():
    # Constants and configuration
    data_dir = r'MEG Plots\Multinomial l2\Silent'
    model_number = 81
    save_fig = False
    labels = ['bird_real', 'cry_real', 'bell_real', 'phone_real']
    epoch_filename = '20210319_SilentDataLabelsEstimates_AR.npy'

    # Load the epochs data from the .npy file
    epochs = load_epochs_data(data_dir, epoch_filename)

    # Extract and transpose the data for the specified model
    best_model = get_model_data(epochs, model_number)

    results_dir = os.path.join(data_dir, str(model_number))
    fig_filename = os.path.join(results_dir, f'EpochVStimepoints_{model_number}.png') if save_fig else None

    # Display or save plot
    create_model_plot(best_model, model_number, True, fig_filename)


if __name__ == '__main__':
    main()
