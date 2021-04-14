import os

import mne
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from os.path import join

plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')

result_file = r'C:\Users\FARAZ\Documents\Git Repositories\MEG_SoundSequences\Silent Results'  # (180, 50, 194, 4)
epochs = np.load(result_file + ".npy", allow_pickle=True)
Models = [68, 81, 92]


for Model in Models:
    best_model = epochs[Model, :, :, :]
    best_model_rs = best_model.transpose(2, 1, 0)  # 4, 194, 50
    best_model_single_label = np.max(best_model_rs, axis=0)

    for number, label in enumerate(best_model_rs):
        best_model_l1 = np.where((label == best_model_single_label), label, 0)

        labels = ['bird_real', 'cry_real', 'bell_real', 'phone_real']
        fig, ax = plt.subplots()
        im = ax.matshow(best_model_l1, cmap='Reds', origin='lower', vmin=0, vmax=1)
        ax.xaxis.set_ticks_position('bottom')
        y = [item.get_text() for item in ax.get_xticklabels()]
        for i, item in enumerate([0, 0, 100, 200, 300, 400, 500]):
            y[i] = item
        ax.set_xticklabels(y)
        ax.set_xlabel('timepoints (ms)')
        ax.set_ylabel('Epochs')
        ax.set_title(f'Accuracy plot of Epochs over timepoints for Model {Model}')
        count = np.count_nonzero(best_model_l1)
        total = 50*194
        plot_info = f"Label: {labels[number]}\nCount: {count} of {total} Ratio: {round(count/total, 3)}"
        plt.figtext(0.52 - 0.1*(1+number), 0.005, plot_info, fontsize=9, va="bottom", ha="left")
        plt.colorbar(im, ax=ax)
        fig.set_size_inches(5, 10)
        plt.show()
        results_dir = join(r"C:\Users\FARAZ\Desktop\20210319", f"{Model}")
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        save_name = join(results_dir, f'{number}')
        plt.savefig(save_name)
