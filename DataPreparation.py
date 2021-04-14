# imports
import mne
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import mne.viz
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')
from os.path import join


data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
fname = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled.fif"
raw_file = join(data_dir, fname)

real_events = {'cry_real_10': 10110, 'bird_real_10': 10100, 'phone_real_10': 10130, 'bell_real_10': 10120}
omitted_events = {'cry_omission_10': 10161, 'phone_omission_10': 10181}

epochs = mne.read_epochs(raw_file, verbose='error')
samplingRate = 100

if 'resampled' not in raw_file:
    epochs.resample(samplingRate, npad='auto')
    print('Data is resampled!')
    raw_file = raw_file[:-4] + '_resampled.fif'
    epochs.save(raw_file)


Perform_Artifacts_Rejection = False


if Perform_Artifacts_Rejection:
    "----> Code for Artifacts Rejection <-------"

    # get epoch ids before excluding anything
    epoch_ids_beforeAR = epochs.selection
    print(len(epoch_ids_beforeAR))

    # Remove artifacts manually
    epochs.plot(scalings='auto')

    epoch_ids_afterAR = epochs.selection
    print(len(epoch_ids_afterAR))

    # Saving Epoch after artifacts removal
    # plot average of epoched data = evoked data
    epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-120, 120]))
    epochs.save(raw_file[:-4] + '_AR.fif')

    rejectedEpoch_ids = [epoch_ids_beforeAR[i] for i in range(len(epoch_ids_beforeAR)) if epoch_ids_beforeAR[i] not in epoch_ids_afterAR]
    print(rejectedEpoch_ids)

    # save the rejected epochOmissionAR for future use
    rejectedEpoch_indices = [np.where(epoch_ids_beforeAR == rejectedEpoch_ids[i])[0][0] for i in range(len(rejectedEpoch_ids))]
    print(len(rejectedEpoch_indices))
    print(rejectedEpoch_indices)
    np.save(raw_file[:-4]+ '_rejectedEpochs_AR', rejectedEpoch_indices)


if not Perform_Artifacts_Rejection:
    "------> Code for Viewing Pre and Post AR MEG data <-------"

    "PRE AR"
    epochs = mne.read_epochs(raw_file, verbose='error')
    epochs[[*real_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))
    epochs[[*omitted_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))

    "POST AR"
    epochsAR = mne.read_epochs(raw_file[:-4]+'_AR.fif', verbose='error')
    epochsAR[[*real_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))
    epochsAR[[*omitted_events]].pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))


