import mne
import numpy as np
import warnings
import matplotlib.pyplot as plt
from os.path import join

# Configure matplotlib and suppress warnings
plt.switch_backend("TkAgg")
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


def load_epochs(data_dir, filenames):
    """
    Load MNE Epochs from the given filenames.

    Parameters:
    - data_dir (str): The directory where the data files are located.
    - filenames (dict): A dictionary mapping keys to filenames.

    Returns:
    - dict: A dictionary mapping keys to loaded Epochs.
    """
    epochs = {}
    for key, fname in filenames.items():
        filepath = join(data_dir, fname)
        epochs[key] = mne.read_epochs(filepath)
    return epochs


def get_evoked(epoch_data, event_types):
    """
    Compute the evoked response for the given event types.

    Parameters:
    - epoch_data (mne.Epochs): The epochs data.
    - event_types (list of str): Event names to include.

    Returns:
    - mne.Evoked: The averaged evoked response.
    """
    return epoch_data[event_types].pick_types(meg='mag').average()


def plot_evoked_joint(evoked, title, window_title, times=None):
    """
    Plot the joint evoked response.

    Parameters:
    - evoked (mne.Evoked): The evoked data to plot.
    - title (str): Title of the plot.
    - window_title (str): Title of the window.
    - times (array-like, optional): Times to plot topomaps at.

    Returns:
    - None
    """
    ts_args = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=window_title)
    topomap_args = dict(vmin=-200, vmax=200)
    mne.viz.plot_evoked_joint(evoked, ts_args=ts_args, topomap_args=topomap_args, title=title, times=times)


def main():
    # MEG datafile directory
    data_dir = r"\ML on MEG\SensoryProcessing_MEG-faraz\MEG_Data"

    # Filenames for different datasets
    filenames = {
        'omission_AR': "17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif",
        'omission': "17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled.fif",
        'silent_AR': "17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif",
        'silent': "17_2_tsss_mc_trans_silent_resampled_100hz_500ms_beforeAR.fif"
    }

    # Load epochs
    epochs = load_epochs(data_dir, filenames)

    # Plot evoked response for all Omission data before AR
    evoked_omission_all = epochs['omission'].pick_types(meg='mag').average()
    evoked_omission_all.plot_joint(ts_args=dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350])))

    # Define event types
    event_types_real = ['cry_real_10', 'bird_real_10', 'phone_real_10', 'bell_real_10']
    event_types_omission = ['cry_omission_10', 'phone_omission_10']
    times = np.array([0.51, 0.59, 0.74])  # Times for topomap plots

    # Plot Real Sounds after AR
    evoked_real_AR = get_evoked(epochs['omission_AR'], event_types_real)
    plot_evoked_joint(evoked_real_AR, title='ERF of Real Sounds after AR', window_title='2 Real Sound AR')

    # Plot Omission Sounds after AR
    evoked_omission_AR = get_evoked(epochs['omission_AR'], event_types_omission)
    plot_evoked_joint(evoked_omission_AR, title='ERF of Omission Sounds after AR', window_title='4 Omission Sound AR',
                      times=times)

    # Plot Real Sounds before AR
    evoked_real_before_AR = get_evoked(epochs['omission'], event_types_real)
    plot_evoked_joint(evoked_real_before_AR, title='ERF of Real Sounds before AR', window_title='1 Real Sound')

    # Plot Omission Sounds before AR
    evoked_omission_before_AR = get_evoked(epochs['omission'], event_types_omission)
    plot_evoked_joint(evoked_omission_before_AR, title='ERF of Omission Sounds before AR',
                      window_title='3 Omission Sound AR', times=times)

    # Silent data before and after AR
    evoked_silent_before_AR = epochs['silent'].pick_types(meg='mag').average()
    evoked_silent_after_AR = epochs['silent_AR'].pick_types(meg='mag').average()

    # Update channel info from evoked_omission_all
    channel_info = evoked_omission_all.info['chs']
    evoked_silent_before_AR.info['chs'] = channel_info
    evoked_silent_after_AR.info['chs'] = channel_info

    # Plot Silent data before AR
    plot_evoked_joint(evoked_silent_before_AR, title='Average of Silent Data before AR', window_title='5 Silent')

    # Plot Silent data after AR
    plot_evoked_joint(evoked_silent_after_AR, title='Average of Silent Data after AR', window_title='6 Silent AR')


if __name__ == "__main__":
    main()
