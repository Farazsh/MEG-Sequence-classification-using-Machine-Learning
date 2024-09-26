import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')


def create_silent_epochs(raw_file):
    """Create epochs from silent MEG data.

    Parameters:
    - raw_file: str, path to the raw FIF file.
    """
    raw = mne.io.read_raw_fif(raw_file, preload=True)

    # Define epoch duration in seconds
    epoch_duration_sec = 0.5  # 500 milliseconds

    # Create fixed-length events
    events = mne.make_fixed_length_events(raw, duration=epoch_duration_sec)

    # Create epochs from the raw data
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=epoch_duration_sec, preload=True)

    # Save the epochs to a FIF file
    base_filename = os.path.splitext(raw_file)[0]
    epochs.save(f"{base_filename}_epoched.fif", overwrite=True)


def resample_silent_epochs(epochs_file, sampling_rate):
    """Resample the silent epochs to a new sampling rate.

    Parameters:
    - epochs_file: str, path to the epochs FIF file.
    - sampling_rate: int or float, desired sampling rate in Hz.
    """
    epochs = mne.read_epochs(epochs_file, preload=True)
    print(f"Old Sampling Frequency: {epochs.info['sfreq']} Hz")

    # Resample the epochs
    epochs_resampled = epochs.copy().resample(sampling_rate, npad='auto')
    print(f"New Sampling Frequency: {epochs_resampled.info['sfreq']} Hz")

    # Save the resampled epochs
    base_filename = os.path.splitext(epochs_file)[0]
    epochs_resampled.save(f"{base_filename}_resampled_{int(sampling_rate)}hz.fif", overwrite=True)


def artifacts_rejection(epochs_file):
    """Perform artifact rejection on the epochs.

    Parameters:
    - epochs_file: str, path to the epochs FIF file.
    """
    # Load epochs
    epochs = mne.read_epochs(epochs_file, preload=True)
    epoch_ids_before_ar = epochs.selection.copy()
    print(f"Number of epochs before artifact rejection: {len(epoch_ids_before_ar)}")

    # Plot the averaged data before artifact rejection
    epochs.pick_types(meg='mag').average().plot(
        scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]),
        title='Average Before Artifact Rejection')

    # Define rejection criteria and perform artifact rejection
    reject_criteria = dict(mag=4e-12)  # Adjust threshold as needed
    epochs.drop_bad(reject=reject_criteria)

    epoch_ids_after_ar = epochs.selection
    print(f"Number of epochs after artifact rejection: {len(epoch_ids_after_ar)}")

    # Identify rejected epochs
    rejected_epoch_ids = np.setdiff1d(epoch_ids_before_ar, epoch_ids_after_ar)
    print(f"Rejected Epoch IDs: {rejected_epoch_ids}")

    # Plot the averaged data after artifact rejection
    epochs.pick_types(meg='mag').average().plot(
        scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]),
        title='Average After Artifact Rejection')

    # Save the epochs after artifact rejection
    base_filename = os.path.splitext(epochs_file)[0]
    epochs.save(f"{base_filename}_afterAR.fif", overwrite=True)

    # Save the indices of rejected epochs for future use
    rejected_epoch_indices = [np.where(epoch_ids_before_ar == epoch_id)[0][0] for epoch_id in rejected_epoch_ids]
    print(f"Indices of Rejected Epochs: {rejected_epoch_indices}")
    np.save(f"{base_filename}_rejectedEpochs_AR.npy", rejected_epoch_indices)


def main():
    # Set data directory and file names
    data_dir = r"MEG\SensoryProcessing_MEG-faraz\MEG_Data"
    fname_raw = "17_2_tsss_mc_trans_silent.fif"
    fname_epoched = "17_2_tsss_mc_trans_silent_epoched.fif"
    fname_resampled = "17_2_tsss_mc_trans_silent_epoched_resampled_100hz.fif"

    raw_file = join(data_dir, fname_raw)
    epochs_file = join(data_dir, fname_epoched)
    resampled_file = join(data_dir, fname_resampled)

    # Flags to control processing steps
    create_epochs_silent = True
    resampling_silent = True
    perform_artifacts_rejection = True

    if create_epochs_silent:
        create_silent_epochs(raw_file)

    if resampling_silent:
        sampling_rate = 100  # Desired sampling rate in Hz
        resample_silent_epochs(epochs_file, sampling_rate)

    if perform_artifacts_rejection:
        artifacts_rejection(resampled_file)


if __name__ == '__main__':
    main()
