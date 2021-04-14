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
plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')


data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"
fname_raw = r"17_2_tsss_mc_trans_silent.fif"
fname_epoched = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_beforeAR.fif"
fname_epoched_AR = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"
raw_file = join(data_dir, fname_raw)

Silent_epochs_file = join(data_dir, fname_epoched)
Silent_epochs_AR_file = join(data_dir, fname_epoched_AR)

Silent_epochs = mne.read_epochs(Silent_epochs_file)

Create_Epochs_Silent = False
Resampling_Silent = False
Perform_Artifacts_Rejection = False

if Create_Epochs_Silent:

    "-------------> Create Epoch from Silent data <-------------------"

    raw = mne.io.read_raw_fif(raw_file)
    raw.load_data()
    milliseconds_in_each_epoch = 500

    events = np.arange(milliseconds_in_each_epoch, 97474, milliseconds_in_each_epoch)
    previous_mark = 0
    epoch_data = []
    raw_data = raw.get_data().T
    for e in events:
        epoch_data.append(raw_data[previous_mark:e].T)
        previous_mark = e
    epoc_data = np.array(epoch_data)

    "###########################################"
    # Useful in case of previous mne versions
    def namw2type(name):
        ch_list = ['eeg', 'meg', 'grad', 'ref_meg', 'chpi', 'sti', 'eog', 'ecg', 'emg', 'seeg', 'bio', 'ecog', 'hbo',
                   'hbr']
        map_dict = {'sti': 'stim', 'meg': 'mag', 'chpi': 'misc'}
        for ch_type in ch_list:
            if ch_type in name.lower():
                return map_dict.get(ch_type, ch_type)
        print(name)
        return 'misc'
    for _meg in raw.ch_names:
        ch_types = [namw2type(_meg) for _meg in raw.ch_names]
    "###########################################"

    info1 = mne.create_info(ch_names=raw.ch_names, ch_types=raw.get_channel_types(), sfreq=raw.info['sfreq'])
    Silent_epochs = mne.EpochsArray(epoc_data, info1)
    Silent_epochs.save(raw_file[:-4] + '_Epoched.fif')

if Resampling_Silent:

    "-----------> Resampling Silent Epoched data <-----------------"

    print(f" Old Frequency : {Silent_epochs.info['sfreq']}")
    print(Silent_epochs.times)
    sampling_rate = 100
    Silent_epochs = Silent_epochs.resample(sampling_rate, npad='auto')
    print(f"New Frequency = {Silent_epochs.info['sfreq']}")
    print(Silent_epochs.times)
    Silent_epochs.save(raw_file[:-4] + f'_resampled_{sampling_rate}hz.fif')

if Perform_Artifacts_Rejection:

    "------------------->  Artifacts Rejection <----------------------------"

    Silent_epochs = mne.read_epochs(Silent_epochs_file)
    epoch_ids_beforeAR = Silent_epochs.selection
    print(len(epoch_ids_beforeAR))
    Silent_epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[-300, 300]))
    epoch_ids_afterAR = Silent_epochs.selection
    print(epoch_ids_afterAR)
    rejectedEpoch_ids = [epoch_ids_beforeAR[i] for i in range(len(epoch_ids_beforeAR)) if epoch_ids_beforeAR[i] not in epoch_ids_afterAR]
    print(rejectedEpoch_ids)
    Silent_epochs.pick_types(meg='mag').average().plot(scalings=dict(mag=1e15), ylim=dict(mag=[--300, 300]))
    Silent_epochs.save(raw_file[:-4]+'_afterAR.fif')

    # save the rejected epochOmissionAR for future use
    rejectedEpoch_indices = [np.where(epoch_ids_beforeAR == rejectedEpoch_ids[i])[0][0] for i in
                             range(len(rejectedEpoch_ids))]
    print(len(rejectedEpoch_indices))
    print(rejectedEpoch_indices)
    np.save(raw_file[:-4] + '_rejectedEpochs_AR', rejectedEpoch_indices)

