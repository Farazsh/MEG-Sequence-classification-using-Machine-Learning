import mne
from mne.decoding import Vectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter
from sklearn.metrics import accuracy_score
import random
import warnings
import matplotlib.pyplot as plt
from os.path import join
import pickle
random.seed(50)
plt.switch_backend("TkAgg")
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


# MEG datafile directory
data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"

fnameOmissionAR = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif"
fnameSilentAR = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"

datafileOmissionAR = join(data_dir, fnameOmissionAR)
datafileSilentAR = join(data_dir, fnameSilentAR)

epochOmissionAR = mne.read_epochs(datafileOmissionAR)
epochSilentAR = mne.read_epochs(datafileSilentAR)

# Results directory
results_dir = r"C:\Users\FARAZ\Desktop\MEG Plots\Results"
results_name = r"MEGAnalysisResult"

# Best Parameter directory
bestParams_dir = r"C:\Users\FARAZ\Desktop\MEG Plots\L2 reg\Best Params"
bestParams_name = r"BestParamsFile.txt"

# Models directory
saved_models = r"C:\Users\FARAZ\Desktop\MEG Plots\Model"
clsfFile = join(saved_models, "2575_predLevel.pkl")
bestParams_file = join(bestParams_dir, bestParams_name)
result_file = join(results_dir, results_name)

epoch_real = epochOmissionAR['bird_real_10', 'phone_real_10', 'bell_real_10', 'cry_real_10']  # only real sounds of interest

Train_Test = False

if Train_Test:
    # Seperate data and label
    data = epoch_real.get_data()
    label = epoch_real.events[:, -1]

    nFolds = 5

    trainData, testData, trainLabel, testLabel = train_test_split(data, label, test_size=0.25, random_state=50, stratify=label)
    print(f"Entire Data: {Counter(label)}")
    print(f"Train Data: {Counter(trainLabel)}")
    print(f"Test Data: {Counter(testLabel)}")


    repeats = nFolds
    rkf = RepeatedKFold(n_splits=nFolds, n_repeats=repeats, random_state=50)

    tlim = 180
    Classifiers_Timepoints = []
    CV_score = np.zeros((repeats * nFolds, tlim))
    train_Score = np.zeros((tlim))
    test_Score = np.zeros((tlim))
    Classifier_list = []
    for tp in np.arange(tlim):
        d2t_cv = trainData[:, :, tp]  # data to test
        d2t_test = testData[:, :, tp]  # data to test - real

        clf = make_pipeline(Vectorizer(), StandardScaler(),
                            LogisticRegression(penalty='l2', multi_class='multinomial'))
        # get CV score:
        CV_score[:, tp] = cross_val_score(clf, d2t_cv, trainLabel, cv=rkf)

        # fit the model using all CV data:
        clf.fit(d2t_cv, trainLabel)
        Classifier_list.append(clf)

        # generalize performance on train data:
        estimated_trainLabel = clf.predict(d2t_cv)
        train_Score[tp] = accuracy_score(trainLabel, estimated_trainLabel, normalize=True)

        # generalize performance on test data:
        estimated_testLabel = clf.predict(d2t_test)
        test_Score[tp] = accuracy_score(testLabel, estimated_testLabel, normalize=True)


    results = [train_Score, test_Score, CV_score]
    np.save(result_file, results)

    with open(clsfFile, "wb") as f:
        for model in Classifier_list:
            pickle.dump(model, f)