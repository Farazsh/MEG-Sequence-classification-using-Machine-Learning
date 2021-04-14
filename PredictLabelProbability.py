# imports
from os.path import join
import mne
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import pickle

trainedModel = r'C:\Users\FARAZ\Desktop\MEG Plots\Multinomial l2\Model\2575_predLevel.pkl'  # Saved Model File
f = open(trainedModel, "r+b")
clfs = []
while 1:
    try:
        clfs.append(pickle.load(f))
    except EOFError:
        break
f.close()

data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
fname_silent = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"
test_file = join(data_dir, fname_silent)

# Real data and labels
epoch_silent = mne.read_epochs(test_file)
silent_data = epoch_silent.get_data()
random.seed(50)
tlim = 50  # timepoints in 1 segment

" -------> Predict Label Probabilities in Silent data <-----------"

classified_list = []  # (180, 50, 194, 4)
for clf in clfs:
    check_list = []
    for tp in np.arange(tlim):
        # generalize performance on test data:
        # print(clf.classes_)
        labels_estim = clf.predict_proba(silent_data[:, :, tp])  # (segments, channels, timepoints)
        check_list.append(labels_estim)
    classified_list.append(check_list)
classified_list = np.array(classified_list)  # (180, 50, 194, 4)


outputfilename = r"C:\Users\FARAZ\Desktop\MEG Plots\Multinomial l2\Silent\20210319_SilentDataLabelsEstimates_AR1.npy"
np.save(outputfilename, classified_list)
