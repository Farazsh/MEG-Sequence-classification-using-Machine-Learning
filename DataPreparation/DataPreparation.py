import mne
import numpy as np
import warnings
from os.path import join
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


def load_epochs(file_path, resample_rate=None):
    """
    Load epochs from a file and optionally resample them.

    :param file_path: Path to the .fif file containing epochs
    :param resample_rate: The sampling rate to resample the epochs, if necessary
    :return: Loaded epochs object
    """
    epochs = mne.read_epochs(file_path, verbose='error')

    if resample_rate and 'resampled' not in file_path:
        epochs.resample(resample_rate, npad='auto')
        print('Data is resampled!')
        save_path = file_path[:-4] + '_resampled.fif'
        epochs.save(save_path)
        return epochs, save_path

    return epochs, file_path


def perform_artifacts_rejection(epochs, save_path):
    """
    Perform artifact rejection on the provided epochs and save the result.

    :param epochs: The epochs to process
    :param save_path: The path to save the processed epochs
    """
    # Store the epoch IDs before rejection
    epoch_ids_before_ar = epochs.selection.copy()
    print(f"Number of epochs before AR: {len(epoch_ids_before_ar)}")

    # Manually remove artifacts
    epochs.plot(scalings='auto')
    epoch_ids_after_ar = epochs.selection.copy()
    print(f"Number of epochs after AR: {len(epoch_ids_after_ar)}")

    # Plot the average of cleaned epochs and save
    epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-120, 120]))
    epochs.save(save_path)

    # Identify rejected epochs
    rejected_epochs = list(set(epoch_ids_before_ar) - set(epoch_ids_after_ar))
    rejected_indices = [np.where(epoch_ids_before_ar == idx)[0][0] for idx in rejected_epochs]

    print(f"Rejected Epoch IDs: {rejected_epochs}")
    print(f"Rejected Epoch Indices: {rejected_indices}")

    # Save rejected epoch indices for future use
    np.save(save_path[:-4] + '_rejectedEpochs_AR', rejected_indices)


def view_pre_post_ar_meg_data(pre_ar_file, real_events, omitted_events):
    """
    View MEG data before and after artifact rejection.

    :param pre_ar_file: Path to the epochs file before AR
    :param real_events: Dictionary of real events
    :param omitted_events: Dictionary of omitted events
    """
    # View PRE AR
    print("Viewing PRE AR data...")
    epochs = mne.read_epochs(pre_ar_file, verbose='error')
    epochs[[*real_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))
    epochs[[*omitted_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))

    # View POST AR
    print("Viewing POST AR data...")
    post_ar_file = pre_ar_file[:-4] + '_AR.fif'
    epochs_ar = mne.read_epochs(post_ar_file, verbose='error')
    epochs_ar[[*real_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))
    epochs_ar[[*omitted_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15),
                                                                      ylim=dict(mag=[-300, 300]))


def main():
    data_dir = r"MEG\SensoryProcessingMEG\MEG_Data"
    fname = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled.fif"
    raw_file = join(data_dir, fname)
    sampling_rate = 100
    perform_ar = False  # Change this to True if you want to perform artifact rejection

    real_events = {
        'cry_real_10': 10110,
        'bird_real_10': 10100,
        'phone_real_10': 10130,
        'bell_real_10': 10120
    }
    omitted_events = {
        'cry_omission_10': 10161,
        'phone_omission_10': 10181
    }

    # Load epochs and resample if necessary
    epochs, current_file_path = load_epochs(raw_file, sampling_rate)

    if perform_ar:
        perform_artifacts_rejection(epochs, current_file_path[:-4] + '_AR.fif')
    else:
        view_pre_post_ar_meg_data(current_file_path, real_events, omitted_events)


if __name__ == "__main__":
    main()
