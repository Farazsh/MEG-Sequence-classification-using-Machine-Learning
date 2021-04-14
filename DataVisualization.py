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

data_dir = r"C:\Users\FARAZ\Documents\Git Repositories\ML on MEG\SensoryProcessing_MEG-faraz\MEG_Data\S17"
fname = r"17_2_tsss_mc_trans_mag_nobase-epochs_afterICA-faraz_resampled_AR1.fif"
raw_file = join(data_dir, fname)

epochs = mne.read_epochs(raw_file)
all_events = epochs.events
mne.viz.plot_events(all_events)

epochs.average().plot()
print(epochs.event_id)

print('Percentage of cry_real_10 events : ', np.around(len(epochs['cry_real_10'])/len(epochs), decimals=2))
print('Percentage of cry_omission_10 events : ', np.around(len(epochs['cry_omission_10'])/len(epochs), decimals=2))
print('Percentage of bird_real_10 events : ', np.around(len(epochs['bird_real_10'])/len(epochs), decimals=2))
print('Percentage of phone_real_10 events : ', np.around(len(epochs['phone_real_10'])/len(epochs), decimals=2))
print('Percentage of phone_omission_10 events : ', np.around(len(epochs['phone_omission_10'])/len(epochs), decimals=2))
print('Percentage of bell_real_10 events : ', np.around(len(epochs['bell_real_10'])/len(epochs), decimals=2))

# epochOmissionAR['cry_real_10'].average().plot()
# epochOmissionAR['cry_omission_10'].average().plot()
# epochOmissionAR['bird_real_10'].average().plot()
# epochOmissionAR['phone_real_10'].average().plot()
# epochOmissionAR['phone_omission_10'].average().plot()
# epochOmissionAR['bell_real_10'].average().plot()


# Plotting using Matplotlib Libraries
cry_r = epochs['cry_real_10']
cry_o = epochs['cry_omission_10']
bird_r = epochs['bird_real_10']
phone_r = epochs['phone_real_10']
phone_o = epochs['phone_omission_10']
bell_r = epochs['bell_real_10']
ch = 18

conditions = ['cry_real_10', 'cry_omission_10', 'bell_real_10', 'phone_real_10', 'phone_omission_10', 'bell_real_10']

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Time Instances')
ax.set_ylabel('Volt')

ax.plot(cry_r.average().data[ch, :], color='blue', label='cry_real_10')
ax.plot(cry_o.average().data[ch, :], color='red', label='cry_omission_10')
ax.plot(bird_r.average().data[ch, :], color='green', label='bird_real_10')
ax.plot(phone_r.average().data[ch, :], color='violet', label='phone_real_10')
ax.plot(phone_o.average().data[ch, :], color='yellow', label='phone_omission_10')
ax.plot(bell_r.average().data[ch, :], color='pink', label='bell_real_10')

legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
plt.title('ERP of different conditions')
# plt.show()

times = np.arange(-0.1, 0.5, 0.1)
epochs.average().plot_topomap(times, ch_type='mag')