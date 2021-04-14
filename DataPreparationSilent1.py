# imports
import mne
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import mne.viz
import warnings

warnings.filterwarnings('ignore')
from os.path import join

mne.set_log_level('WARNING')

data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
fname = r"17_2_tsss_mc_trans_silent.fif"
raw_file = join(data_dir, fname)

rejected_epochs = np.load(join(r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data\Silent_500", "17_2_tsss_mc_trans_silent_rejectedEpochs_AR.npy"))

simulated_epochs = mne.read_epochs(join(data_dir, r"17_2_tsss_mc_trans_silent_beforeAR.fif"))
epoch_ids_beforeAR = simulated_epochs.selection

simulated_epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-200, 200]))
simulated_epochs.plot()
simulated_epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-200, 200]))
simulated_epochs.save(raw_file[:-4]+'_afterAR.fif')
epoch_ids_afterAR = simulated_epochs.selection
rejectedEpoch_ids = [epoch_ids_beforeAR[i] for i in range(len(epoch_ids_beforeAR)) if epoch_ids_beforeAR[i] not in epoch_ids_afterAR]
# save the rejected epochOmissionAR for future use
rejectedEpoch_indices = [np.where(epoch_ids_beforeAR == rejectedEpoch_ids[i])[0][0] for i in range(len(rejectedEpoch_ids))]
print(len(rejectedEpoch_indices))
print(rejectedEpoch_indices)
np.save(raw_file[:-4]+'_rejectedEpochs_AR', rejectedEpoch_indices)
