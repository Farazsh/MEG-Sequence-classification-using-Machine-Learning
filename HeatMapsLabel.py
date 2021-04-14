import os

import mne
import numpy as np
import warnings

from matplotlib import ticker, cm
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from os.path import join

plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')
epochs = np.load(r'C:\Users\FARAZ\Desktop\MEG Plots\Multinomial l2\Silent\20210319_SilentDataLabelsEstimates_AR.npy',
                 allow_pickle=True)

Model = 81
best_model = epochs[Model, :, :].transpose()

# best_model_rs = best_model.reshape(4, 194, 50)
# best_model_rs1 = best_model.sum(axis=2)
# for model in best_model:
#     for tp in model:
#         print(f'{tp}')
# best_model_rs = best_model.transpose(2, 1, 0)  # 4, 194, 50
# best_model_single_label = np.max(best_model_rs, axis=0)
# best_model_rs1 = best_model_rs.sum(axis=0)
# for model in best_model_rs:
#     for tp in model:
#         print(f'{tp}')
# max_val = np.max(epochOmissionAR)
# model_tp = np.where(epochOmissionAR==max_val)
# print(max_val)
# print(model_tp)
# for ep, epoch in enumerate(epochOmissionAR):
# for number, label in enumerate(best_model_rs):
#     Old function Modified
#     best_model_l11 = np.zeros((194, 50), float)
#     for _l , _label in enumerate(label):
#         for _e, epoch in enumerate(_label):
#             if _label == best_model_single_label[_l][_e]
#                 best_model_l11[_l][_e] = epoch
# best_model_l1 = np.where((label == best_model_single_label).all(axis=1, keepdims=True), label, 0)

labels = ['bird_real', 'cry_real', 'bell_real', 'phone_real']
fig, ax = plt.subplots()
im = ax.matshow(best_model, origin='lower')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('timepoints (secs)')
ax.set_ylabel('Epochs')
ax.set_title(f'Performance of Model_{Model} over 50 timepoints for 194 Epochs')
cb = plt.colorbar(im, ax=ax)
fig.set_size_inches(8, 25)
results_dir = join(r"C:\Users\FARAZ\Desktop\MEG Plots\Multinomial l2\Silent", str(Model))
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
plt.show()
# plt.savefig(
#     join(results_dir, f'EpochVStimepoints_{Model}_{labels[number]}'))
