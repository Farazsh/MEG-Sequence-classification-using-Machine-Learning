# Imports
from os.path import join
import mne
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle


def load_trained_models(filename):
    """Load trained classifiers from a pickle file.

    Parameters:
    filename (str): Path to the pickle file containing trained classifiers.

    Returns:
    list: A list of trained classifiers.
    """
    clfs = []
    with open(filename, "rb") as f:
        while True:
            try:
                clfs.append(pickle.load(f))
            except EOFError:
                break
    return clfs


def load_silent_data(data_dir, fname_silent):
    """Load silent data from an MNE epochs file.

    Parameters:
    data_dir (str): Directory containing the data file.
    fname_silent (str): Filename of the silent data epochs file.

    Returns:
    numpy.ndarray: Array containing the silent data.
    """
    test_file = join(data_dir, fname_silent)
    epoch_silent = mne.read_epochs(test_file)
    silent_data = epoch_silent.get_data()
    return silent_data


def predict_label_probabilities(clfs, silent_data, tlim):
    """
    Predict label probabilities for silent data using trained classifiers.

    Parameters:
    clfs (list): List of trained classifiers.
    silent_data (numpy.ndarray): Silent data array of shape (n_samples, n_channels, n_times).
    tlim (int): Number of time points to consider.

    Returns:
    numpy.ndarray: Array of predicted label probabilities with shape (n_clfs, tlim, n_samples, n_classes).
    """
    classified_list = []
    for clf in clfs:
        check_list = []
        for tp in range(tlim):
            # Predict probabilities for each time point
            labels_estim = clf.predict_proba(silent_data[:, :, tp])
            check_list.append(labels_estim)
        classified_list.append(check_list)
    classified_list = np.array(classified_list)
    return classified_list


def save_classified_list(output_filename, classified_list):
    """Save the classified list to a numpy file.

    Parameters:
    output_filename (str): Path to the output file.
    classified_list (numpy.ndarray): Array of predicted label probabilities.
    """
    np.save(output_filename, classified_list)


def main():
    # Load trained classifiers
    trained_model_path = r'MEG Plots\Multinomial l2\Model\2575_predLevel.pkl'
    clfs = load_trained_models(trained_model_path)

    # Load silent data
    data_dir = r"ML on MEGSensoryProcessing_MEG-faraz\MEG_Data"
    fname_silent = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"
    silent_data = load_silent_data(data_dir, fname_silent)

    # Time points limit
    tlim = 50  # Number of time points in one segment

    # Predict label probabilities
    classified_list = predict_label_probabilities(clfs, silent_data, tlim)

    # Save the classified list
    output_filename = r"MEG Plots\Multinomial l2\Silent\20210319_SilentDataLabelsEstimates_AR1.npy"
    save_classified_list(output_filename, classified_list)


if __name__ == "__main__":
    main()
