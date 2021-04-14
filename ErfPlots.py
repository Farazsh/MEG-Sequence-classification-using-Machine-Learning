import mne
import numpy as np
import warnings
import matplotlib.pyplot as plt
from os.path import join
plt.switch_backend("TkAgg")
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


# MEG datafile directory
data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\check\SensoryProcessing_MEG-faraz\MEG_Data"

fnameOmissionAR = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR.fif"
fnameOmission = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled.fif"

fnameSilentAR = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_afterAR.fif"
fnameSilent = r"17_2_tsss_mc_trans_silent_resampled_100hz_500ms_beforeAR.fif"

datafileOmissionAR = join(data_dir, fnameOmissionAR)
datafileOmission = join(data_dir, fnameOmission)
datafileSilentAR = join(data_dir, fnameSilentAR)
datafileSilent = join(data_dir, fnameSilent)

epochOmissionAR = mne.read_epochs(datafileOmissionAR)
epochOmission = mne.read_epochs(datafileOmission)
evokedOmissionALL = epochOmission.pick_types(meg='mag').average()
evokedOmissionALL.plot_joint(ts_args=dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350])))

# # Real and Omission parts in omission trails after AR
times = np.array([0.51, 0.59, 0.74])
win_title = '2 Real Sound AR'
title = 'ERF of Real Sounds after AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)
TOPOMAP_args = dict(vmin=-200, vmax=200)

evokedOmissionARreal = epochOmissionAR[['cry_real_10', 'bird_real_10', 'phone_real_10', 'bell_real_10']].pick_types(meg='mag').average()
mne.viz.plot_evoked_joint(evokedOmissionARreal, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title)

win_title = '4 Omission Sound AR'
title = 'ERF of Omission Sounds after AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)

evokedOmissionARomission = epochOmissionAR[['cry_omission_10', 'phone_omission_10']].pick_types(meg='mag').average()
mne.viz.plot_evoked_joint(evokedOmissionARomission, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title, times=times)

win_title = '1 Real Sound'
title = 'ERF of Real Sounds before AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)

evokedOmissionReal2 = epochOmission[['cry_real_10', 'bird_real_10', 'phone_real_10', 'bell_real_10']].pick_types(meg='mag').average()
mne.viz.plot_evoked_joint(evokedOmissionReal2, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title)


times = np.array([0.51, 0.59, 0.74])
win_title = '3 Omission Sound AR'
title = 'ERF of Omission Sounds before AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)

evokedOmissionARomission2 = epochOmission[['cry_omission_10', 'phone_omission_10']].pick_types(meg='mag').average()
mne.viz.plot_evoked_joint(evokedOmissionARomission2, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title, times=times)


# Silent data before and after AR
epochSilent = mne.read_epochs(datafileSilent)
epochSilentAR = mne.read_epochs(datafileSilentAR)
evokedSilent = epochSilent.pick_types(meg='mag').average()
evokedSilentAR = epochSilentAR.pick_types(meg='mag').average()
channel_info = evokedOmissionALL.info['chs']
evokedSilent.info['chs'] = channel_info
evokedSilentAR.info['chs'] = channel_info


win_title = '5 Silent'
title = 'Average of Silent Data before AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)
mne.viz.plot_evoked_joint(evokedSilent, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title)

win_title = '6 Silent AR'
title = 'Average of Silent Data after AR'
TS_ARGS = dict(scalings=dict(mag=1e15), ylim=dict(mag=[-350, 350]), window_title=win_title)
mne.viz.plot_evoked_joint(evokedSilentAR, ts_args=TS_ARGS, topomap_args=TOPOMAP_args, title=title)