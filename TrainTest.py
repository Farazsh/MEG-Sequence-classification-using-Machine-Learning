import os
import pickle
import warnings
from collections import Counter

import mne
import numpy as np
from mne.decoding import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


def load_data(data_dir, fname_omission_ar, fname_silent_ar):
    """
    Load MEG epochs data from specified files.

    Parameters:
    data_dir (str): Directory where data files are located.
    fname_omission_ar (str): Filename for omission AR epochs.
    fname_silent_ar (str): Filename for silent AR epochs.

    Returns:
    tuple: epoch_omission_ar, epoch_silent_ar
    """
    datafile_omission_ar = os.path.join(data_dir, fname_omission_ar)
    datafile_silent_ar = os.path.join(data_dir, fname_silent_ar)
    epoch_omission_ar = mne.read_epochs(datafile_omission_ar)
    epoch_silent_ar = mne.read_epochs(datafile_silent_ar)
    return epoch_omission_ar, epoch_silent_ar


def split_data(epochs, test_size=0.25, random_state=50):
    """
    Split the epochs data into training and testing sets.

    Parameters:
    epochs (mne.Epochs): Epochs data to split.
    test_size (float): Proportion of data to include in the test split.
    random_state (int): Seed used by the random number generator.

    Returns:
    tuple: train_data, test_data, train_labels, test_labels
    """
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return train_data, test_data, train_labels, test_labels


def train_and_evaluate(
    train_data, train_labels, test_data, test_labels,
    n_folds=5, n_repeats=5, n_time_points=180, random_state=50
):
    """
    Train classifiers for each time point and evaluate their performance.

    Parameters:
    train_data (numpy.ndarray): Training data of shape (n_samples, n_features, n_time_points).
    train_labels (numpy.ndarray): Training labels.
    test_data (numpy.ndarray): Testing data of shape (n_samples, n_features, n_time_points).
    test_labels (numpy.ndarray): Testing labels.
    n_folds (int): Number of folds for cross-validation.
    n_repeats (int): Number of repeats for cross-validation.
    n_time_points (int): Number of time points to evaluate.
    random_state (int): Seed used by the random number generator.

    Returns:
    tuple: train_scores, test_scores, cv_scores, classifier_list
    """
    rkf = RepeatedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=random_state
    )

    cv_scores = np.zeros((n_folds * n_repeats, n_time_points))
    train_scores = np.zeros(n_time_points)
    test_scores = np.zeros(n_time_points)
    classifier_list = []

    for tp in range(n_time_points):
        # Extract data for the current time point
        train_data_tp = train_data[:, :, tp]
        test_data_tp = test_data[:, :, tp]

        # Define the classification pipeline
        clf = make_pipeline(
            Vectorizer(),
            StandardScaler(),
            LogisticRegression(
                penalty='l2',
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000
            )
        )

        # Perform cross-validation and store the scores
        cv_scores[:, tp] = cross_val_score(
            clf, train_data_tp, train_labels, cv=rkf
        )

        # Fit the classifier on the entire training data
        clf.fit(train_data_tp, train_labels)
        classifier_list.append(clf)

        # Predict on the training data and calculate training accuracy
        predicted_train_labels = clf.predict(train_data_tp)
        train_scores[tp] = accuracy_score(train_labels, predicted_train_labels)

        # Predict on the testing data and calculate testing accuracy
        predicted_test_labels = clf.predict(test_data_tp)
        test_scores[tp] = accuracy_score(test_labels, predicted_test_labels)

    return train_scores, test_scores, cv_scores, classifier_list


def save_results(results, result_file):
    """
    Save the results to a NumPy file.

    Parameters:
    results (list): List of results to save.
    result_file (str): Filename to save the results to.
    """
    np.save(result_file, results)


def save_classifiers(classifier_list, classifier_file):
    """
    Save the trained classifiers to a file.

    Parameters:
    classifier_list (list): List of trained classifiers.
    classifier_file (str): Filename to save the classifiers to.
    """
    with open(classifier_file, "wb") as f:
        pickle.dump(classifier_list, f)


def main():
    seed = 25
    np.random.seed(seed)

    # File and directory paths
    data_dir = r"\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
    results_dir = r"\MEG Plots\Results"
    saved_models_dir = r"\MEG Plots\Model"

    fname_omission_ar = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif"
    fname_silent_ar = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"

    # Load epochs data
    epoch_omission_ar, epoch_silent_ar = load_data(data_dir, fname_omission_ar, fname_silent_ar)

    # Select real sounds of interest from the omission epochs
    epoch_real = epoch_omission_ar['bird_real_10', 'phone_real_10', 'bell_real_10', 'cry_real_10']

    # Split data into train and test sets
    train_data, test_data, train_labels, test_labels = split_data(epoch_real)

    # Display counts of labels in each set
    print(f"Entire Data: {Counter(epoch_real.events[:, -1])}")
    print(f"Train Data: {Counter(train_labels)}")
    print(f"Test Data: {Counter(test_labels)}")

    # Train and evaluate models
    train_scores, test_scores, cv_scores, classifier_list = train_and_evaluate(
        train_data, train_labels, test_data, test_labels)

    # Save results and classifiers
    results_name = "MEGAnalysisResult.npy"
    classifier_filename = os.path.join(saved_models_dir, "2575_predLevel.pkl")
    result_file = os.path.join(results_dir, results_name)

    results = [train_scores, test_scores, cv_scores]
    save_results(results, result_file)
    save_classifiers(classifier_list, classifier_filename)


if __name__ == "__main__":
    main()
