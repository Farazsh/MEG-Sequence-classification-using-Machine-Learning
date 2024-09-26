import mne
from mne.decoding import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter
from os.path import join
import random
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    A class to load MEG data from a .fif file and preprocess it.

    Attributes:
        data_dir (str): The directory containing the data file.
        fname (str): The name of the data file.
        data_file (str): The full path to the data file.
        epochs (mne.Epochs): The loaded MEG epochs.
        data (np.ndarray): The data array extracted from epochs.
        labels (np.ndarray): The labels corresponding to the data.
    """

    def __init__(self, data_dir, fname):
        """
        Initialize the DataLoader with the data directory and file name.

        Parameters:
            data_dir (str): The directory containing the data file.
            fname (str): The name of the data file.
        """
        self.data_dir = data_dir
        self.fname = fname
        self.data_file = join(self.data_dir, self.fname)
        self.epochs = None
        self.data = None
        self.labels = None

    def load_data(self):
        """
        Load MEG epochs data from the file and select specific events.

        Returns:
            data (np.ndarray): The data array extracted from epochs.
            labels (np.ndarray): The labels corresponding to the data.
        """
        # Read epochs from file
        self.epochs = mne.read_epochs(self.data_file)

        # Select specific events (real sounds of interest)
        self.epochs = self.epochs['cry_real_10', 'bird_real_10', 'phone_real_10', 'bell_real_10']

        # Get data and labels
        self.data = self.epochs.get_data()
        self.labels = self.epochs.events[:, -1]

        return self.data, self.labels


class ModelTrainer:
    """
    A class to train a machine learning model using GridSearchCV.

    Attributes:
        n_folds (int): The number of folds for cross-validation.
        random_state (int): The random state for reproducibility.
        pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
        best_params (dict): The best parameters found by GridSearchCV.
    """

    def __init__(self, n_folds=5, random_state=50):
        """
        Initialize the ModelTrainer with the number of folds and random state.

        Parameters:
            n_folds (int): The number of folds for cross-validation.
            random_state (int): The random state for reproducibility.
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None

    def train(self, X_train, y_train):
        """
        Train the model with GridSearchCV and find the best parameters.

        Parameters:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.

        Returns:
            pipeline (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
        """
        repeats = self.n_folds
        rkf = RepeatedKFold(n_splits=self.n_folds, n_repeats=repeats, random_state=self.random_state)

        # Define parameter grid for Logistic Regression
        LRparam_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'max_iter': list(range(100, 800, 100)),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }

        # Set up the pipeline with vectorization, scaling, and grid search
        clf = LogisticRegression(n_jobs=-1)
        grid_search = GridSearchCV(clf, LRparam_grid, cv=rkf)
        self.pipeline = make_pipeline(Vectorizer(), StandardScaler(), grid_search)

        # Fit the pipeline on the training data
        self.pipeline.fit(X_train, y_train)

        # Retrieve best parameters from GridSearchCV
        self.best_params = self.pipeline.named_steps['gridsearchcv'].best_params_

        return self.pipeline


class ModelEvaluator:
    """
    A class to evaluate the trained model on test data.

    Attributes:
        model (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
    """

    def __init__(self, model):
        """
        Initialize the ModelEvaluator with the trained model.

        Parameters:
            model (sklearn.pipeline.Pipeline): The trained machine learning pipeline.
        """
        self.model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Parameters:
            X_test (np.ndarray): The test data.
            y_test (np.ndarray): The test labels.

        Returns:
            score (float): The accuracy score of the model on the test data.
        """
        score = self.model.score(X_test, y_test)
        return score


class ParameterSaver:
    """
    A class to save the best parameters to a file.

    Attributes:
        params (dict): The best parameters.
        file_path (str): The path to the file where parameters will be saved.
    """

    def __init__(self, params, file_path):
        """
        Initialize the ParameterSaver with parameters and file path.

        Parameters:
            params (dict): The best parameters.
            file_path (str): The path to the file where parameters will be saved.
        """
        self.params = params
        self.file_path = file_path

    def save(self):
        """
        Save the best parameters to the file.
        """
        with open(self.file_path, 'w') as file:
            for key, value in self.params.items():
                file.write(f'{key}: {value}\n')


def main():
    """
    The main function to execute data loading, model training, evaluation, and saving the best parameters.
    """
    random.seed(50)

    # Define file paths
    data_dir = r"\ML on MEG\SensoryProcessing_MEG-faraz\MEG_Data"
    fname = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif"
    best_params_dir = r"\MEG Plots\Parameter"
    best_params_name = r"BestParamsFile.txt"
    best_params_file = join(best_params_dir, best_params_name)

    # Load data
    data_loader = DataLoader(data_dir, fname)
    data, labels = data_loader.load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=50, stratify=labels)

    # Display data distribution
    print(f"Entire Data: {Counter(labels)}")
    print(f"Train Data: {Counter(y_train)}")
    print(f"Test Data: {Counter(y_test)}")

    # Train the model
    model_trainer = ModelTrainer(n_folds=5, random_state=50)
    model = model_trainer.train(X_train, y_train)

    # Evaluate the model
    evaluator = ModelEvaluator(model)
    score = evaluator.evaluate(X_test, y_test)
    print(f"Test Score: {score}")
    print("Best Parameters:")
    for key, value in model_trainer.best_params.items():
        print(f"{key}: {value}")

    # Save best parameters to file
    param_saver = ParameterSaver({'best penalty': model_trainer.best_params['penalty']}, best_params_file)
    param_saver.save()


if __name__ == '__main__':
    main()
