import mne
from mne import find_events
from mne.decoding import Vectorizer, SlidingEstimator, cross_val_multiscore, GeneralizingEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import Counter
import random
import warnings

warnings.filterwarnings('ignore')
import import_ipynb
import matplotlib
import matplotlib.pyplot as plt
import pickle

random.seed(50)
from os.path import join

plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')

# MEG datafile directory
data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
fname = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif"
data_file = join(data_dir, fname)

# Results directory
results_dir = r"C:\Users\FARAZ\Desktop\MEG Plots\Results"
results_name = r"MEGAnalysisResult"

# Best Parameter directory
bestParams_dir = r"C:\Users\FARAZ\Desktop\MEG Plots\Parameter"
bestParams_name = r"BestParamsFile.txt"

# Models directory
saved_models = r"C:\Users\FARAZ\Desktop\MEG Plots\Model"
clsfFile = join(saved_models, "2575_predLevel.pkl")
bestParams_file = join(bestParams_dir, bestParams_name)
result_file = join(results_dir, results_name)

random.seed(50)
tlim = 180
epoch = mne.read_epochs(data_file)  # read epochOmissionAR

epoch_real = epoch['cry_real_10', 'bird_real_10', 'phone_real_10', 'bell_real_10']  # only real sounds of interest

# Seperate data and label
data = epoch_real.get_data()
label = epoch_real.events[:, -1]

# le=LabelEncoder()
# labels_real_binarized=le.fit_transform(label)
nFolds = 5

trainData, testData, trainLabel, testLabel = train_test_split(data, label, test_size=0.25, random_state=50,
                                                              stratify=label)
print(f"Entire Data: {Counter(label)}")
print(f"Train Data: {Counter(trainLabel)}")
print(f"Test Data: {Counter(testLabel)}")

repeats = nFolds
rkf = RepeatedKFold(n_splits=nFolds, n_repeats=repeats, random_state=50)

# 2. make optimization pipeline:
parameters = {'penalty': ['l1', 'l2']}
LRparam_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'max_iter': list(range(100, 800, 100)),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
clf_opt = make_pipeline(Vectorizer(), StandardScaler(),
                        GridSearchCV(LogisticRegression(n_jobs=-1), LRparam_grid, cv=rkf))

# 3. Use CV data to fit and optimize our classifier:
clf_opt.fit(trainData, trainLabel)

# 4. retrieve optimal parameters:
tmp = clf_opt.steps[-1][1]
best_penalty = tmp.best_params_['penalty']
c_value = tmp.best_params_['C']
solver = tmp.best_params_['solver']
max_iter = tmp.best_params_['max_iter']

# 5. Use the optimized classifier on the test dataset (w/o time):
score = clf_opt.score(testData, testLabel)

print(score)
print(f'Best Penalty: {best_penalty}\nC : {c_value}\nSolver: {solver}\nMax Iter:{max_iter}')

# save the best params for later use
file = open(bestParams_file, "w")
file.writelines('best penalty: ' + best_penalty)
file.close()
