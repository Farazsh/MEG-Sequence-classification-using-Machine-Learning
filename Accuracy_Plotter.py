import mne
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from patsy.mgcv_cubic_splines import cr
random.seed(42)
from os.path import join
plt.switch_backend("TkAgg")
mne.set_log_level('WARNING')
from scipy.stats import sem


def plot_smooth_curve(df, y1, y2, y3):

    x = np.arange(-500, 1300, 10)

    # Generate spline basis with different degrees of freedom
    x_basis = cr(x, df=df, constraints="center")
    # Fit model to the data
    model1 = LinearRegression().fit(x_basis, y1)
    model2 = LinearRegression().fit(x_basis, y2)

    # Get estimates
    y_hat1 = model1.predict(x_basis)
    y_hat2 = model2.predict(x_basis)

    # Repeat above steps for 25 times and y3 is (25, 180) and then take mean =|
    y_hat3 = []
    for _y3 in y3:
        model3 = LinearRegression().fit(x_basis, _y3)
        y_hat31 = model3.predict(x_basis)
        y_hat3.append(y_hat31)
    y_hat3 = np.array(y_hat3)
    y_hat3m = y_hat3.mean(axis=0)

    # Reference line
    yref = [0.25 for _ in x_basis]

    ax = plt.axes()
    ax.plot(x, yref, '--',  color='#b2abd2', label='theoritical\nchance level')
    ax.plot(x, y_hat2, color='#5e3c99', label='Test')

    # ax.plot(x, y_hat1, color='r', label='Train')
    # ax.plot(x, y_hat3m, color='#B7950B', label='CV')

    # plotting std around cross-validation curve
    # al = 0.4
    # std = sem(y3, axis=0)
    # plt.fill_between(x, y_hat3m, y_hat3m + std, color='y', interpolate=False, alpha=al, label='st dev')
    # plt.fill_between(x, y_hat3m, y_hat3m - std, color='y', interpolate=False, alpha=al)



    plt.ylim(0.15, 0.6)
    plt.xlim(-500, 1300)
    plt.xlabel('time (ms)')
    plt.ylabel('accuracy')
    # plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize='small')
    # plt.legend(loc='upper left', fontsize='small')
    plt.title(f"Classification Performance over time")

    # Annotating max accuracy
    xmax = x[np.argmax(y_hat2)]
    ymax = y_hat2.max()
    annot_max(xmax, ymax)

    # Annotating max accuracy
    _x = x[90:]
    _y1 = y_hat2[90:]
    xmax = _x[np.argmax(_y1)]
    ymax = _y1.max()
    annot_max2(xmax, ymax, ax)

    # Annotating max accuracy
    _x = x[:70]
    _y1 = y_hat2[:70]
    xmax = _x[np.argmax(_y1)]
    ymax = _y1.max()
    # annot_max3(xmax, ymax, ax)

    # plt.savefig(join(r"C:\Users\FARAZ\Desktop\NLP", str("%02d" % df)+'_smoothed_new1'))
    plt.show()
    plt.close("all")


def plot_base_curve(y1, y2, y3):
    ax = plt.axes()
    # x = np.arange(-0.5, 1.3, 0.01)
    x = np.arange(-500, 1300, 10)

    y3m = y3.mean(axis=0)
    # Reference line
    yref = [0.25 for _ in x]

    ax.plot(x, y1, color='r', label='Train')
    ax.plot(x, y2, color='#b2abd2', label='Test')
    ax.plot(x, y3m, color='#fdb863', label='CV')
    ax.plot(x, yref, color='#5e3c99', label='theoritical\nchance level')


    # plotting std around cross-validation curve
    al = 0.3
    std = sem(y3, axis=0) / 2
    plt.fill_between(x, y3m, y3m + std, color='y', interpolate=False, alpha=al, label='std dev')
    plt.fill_between(x, y3m, y3m - std, color='y', interpolate=False, alpha=al)

    plt.ylim(0.13, 0.88)
    plt.xlim(-500, 1300)
    plt.xlabel('time (ms)')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(loc='upper left', fontsize='small')
    plt.title(f"Classification Performance over time")
    savename = 'Base_Curve_Max_acc_'

    # Annotating max accuracy
    # frange = 75
    # x = x[:75]
    # y1 = y1[:75]
    # xmax = x[np.argmax(y1)]
    # ymax = y1.max()
    # annot_max(xmax, ymax)

    # Annotating max accuracy
    xmax = x[np.argmax(y1)]
    ymax = y1.max()
    annot_max(xmax, ymax)

    # Annotating max accuracy
    _x = x[90:]
    _y1 = y1[90:]
    xmax = _x[np.argmax(_y1)]
    ymax = _y1.max()
    annot_max2(xmax, ymax, ax)

    # plt.show()
    plt.savefig(join(r"C:\Users\FARAZ\Desktop\NLP", savename+'new_color'))
    plt.close("all")


def annot_max(xmax, ymax, ax=None):
    text= "Time: {}ms, acc={:.4f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.95, 0.85), **kw)


def annot_max2(xmax, ymax, ax=None):
    text= "Time: {}ms, acc={:.4f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.98, 0.65), **kw)


def annot_max3(xmax, ymax, ax=None):
    text= "Time: {}ms, acc={:.4f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.05, 0.75), **kw)


results_dir = r"C:\Users\FARAZ\Desktop\MEG Plots\Multinomial l2\Results"
results_name = r"MEGAnalysisResult"
result_file = join(results_dir, results_name)
#
results_9 = np.load(result_file+".npy", allow_pickle=True)

# plot_base_curve(results_9[0], results_9[1], results_9[2])

# for dfs in range(35, 50):
plot_smooth_curve(46, results_9[0], results_9[1], results_9[2])