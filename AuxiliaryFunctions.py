# imports
import mne
from os import path
from mne import find_events
from mne.decoding import Vectorizer, SlidingEstimator, cross_val_multiscore

import numpy as np
from patsy.mgcv_cubic_splines import cr

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import pickle
random.seed(42)
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from CustomReporter import class_report


def loadData(s_id, sensors, fname, resampled=False):
    if resampled:
        epochs = mne.read_epochs(fname, verbose='error')
        print(fname + ' loaded!')
    else:
        epochs = mne.read_epochs(fname, verbose='error')
        resampleData(100, epochs, fname)

    return epochs


def resampleData(samplingRate, epochs, filename):
    epochs.resample(samplingRate, npad='auto')
    fname = filename[:-4] + '_resampled.fif'
    epochs.save(fname)
    print(fname + ' loaded!')
    return epochs


def extractDataAndLabels(epochs, eventIdsList):
    data, labels = [], []
    # print(labels)
    # real livings by level
    for l in eventIdsList:
        # print(l)
        if l in epochs.event_id:
            epochs_tmp = epochs[l]
            data_tmp = epochs_tmp.get_data()
            # print(len(data_tmp))
            data.append(data_tmp)
            if 'living' in l:
                labels_tmp = np.zeros(data_tmp.shape[0])
            else:
                labels_tmp = np.ones(data_tmp.shape[0])

            labels.append(labels_tmp)
        else:
            data.append([])
            labels.append([])
    return data, labels


def extractDataAndLabels4class(epochs, eventIdsList):
    data, labels = [], []
    # print(labels)
    # real livings by level
    for l in eventIdsList:
        # print(l)
        if l in epochs.event_id:
            epochs_tmp = epochs[l]
            data_tmp = epochs_tmp.get_data()
            # print(len(data_tmp))
            data.append(data_tmp)
            if 'real' in l:
                labels_tmp = np.zeros(data_tmp.shape[0])
            else:
                labels_tmp = np.ones(data_tmp.shape[0])

            labels.append(labels_tmp)
        else:
            data.append([])
            labels.append([])
    return data, labels


def autolabel(rects, ax, isFloat):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if isFloat is True:
            ax.text(rect.get_x() + rect.get_width()/2., 1.002*height, '%.2f' % float(height), ha='center', va='bottom')

        else:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.002 * height, '%d' % int(height), ha='center', va='bottom')


def plotEvalMetrics(tasks, metricNames, results, filename, outputfilename):
    print(len(results[0]))
    width = 0.15
    fig, ax = plt.subplots(figsize=(15, 10))
    # Set position of bar on X axis
    rects = []
    if len(tasks) > 1:
        width_btw = width
    else:
        width_btw = 2 * width

    rects1 = np.arange(len(tasks))
    if len(results[0]) > 1:
        rects2 = [x + width_btw for x in rects1]
        if len(results[0]) > 2:
            rects3 = [x + width_btw for x in rects2]
            if len(results[0]) > 3:
                rects4 = [x + width_btw for x in rects3]
                if len(results[0]) > 4:
                    rects5 = [x + width_btw for x in rects4]
                    if len(results[0]) > 5:
                        rects6 = [x + width_btw for x in rects5]

    if len(tasks) == 1:
        labels = [None for i in range(len(metricNames))]
    else:
        labels = metricNames
    rects.append(
        ax.bar(rects1, list(zip(*results))[0], color='#87CEFA', width=width, edgecolor='white', label=metricNames[0]))
    if len(results[0]) > 1:
        rects.append(ax.bar(rects2, list(zip(*results))[1], color='#FFE4E1', width=width, edgecolor='white',
                            label=metricNames[1]))
        if len(results[0]) > 2:
            rects.append(ax.bar(rects3, list(zip(*results))[2], color='#CD5C5C', width=width, edgecolor='white',
                                label=metricNames[2]))
            if len(results[0]) > 3:
                rects.append(ax.bar(rects4, list(zip(*results))[3], color='#D1F9F0', width=width, edgecolor='white',
                                    label=metricNames[3]))
                if len(results[0]) > 4:
                    rects.append(ax.bar(rects5, list(zip(*results))[4], color='#6C5B7B', width=width, edgecolor='white',
                                        label=metricNames[4]))
                    if len(results[0]) > 5:
                        rects.append(
                            ax.bar(rects6, list(zip(*results))[5], color='#99B898', width=width, edgecolor='white',
                                   label=metricNames[5]))

    plt.axhline(0.5, color='black', label='50%')

    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.xlabel('Classification Tasks')

    if len(tasks) > 1:
        distance = (len(results) * width) / 2
        plt.xticks([r + distance for r in range(len(results))], tasks)
    else:
        # distance = width/4
        plt.xticks([rect[0].get_x() + rect[0].get_width() / 2. for rect in rects], metricNames)
        plt.rcParams['xtick.labelsize'] = 8
    plt.ylabel('Score')
    plt.title('LR Performance')

    if len(tasks) > 1:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', )
    for r in rects: autolabel(r, ax, isFloat=True)
    plt.savefig(filename, bbox_inches='tight', dpi=80)
    plt.show()

    np.save(outputfilename, results)


def getBestParams(filename):
    params, words = [], []

    with open(filename, 'r') as file:
        for line in file:
            words.append(line.split())

    for i in range(len(words)):
        params.append(words[i][-1])

    return params


def trainAndTest_MVPA(data_real, labels_real, test_data, test_labels, outputfilename, bestParametersFile, tlim,
                      modelsFile, nFolds=5, bestParamsFound=True):
    train_data_real, test_data_real, train_labels_real, test_labels_real = train_test_split(data_real, labels_real,
                                                                                            test_size=0.25,
                                                                                            random_state=42)
    repeats = nFolds
    rkf = RepeatedKFold(n_splits=nFolds, n_repeats=repeats, random_state=42)

    if not bestParamsFound:
        # 2. make optimization pipeline:

        parameters = {'penalty': ['l1', 'l2']}
        clf_opt = make_pipeline(Vectorizer(), StandardScaler(), GridSearchCV(LogisticRegression(), parameters, cv=rkf))

        # 3. Use CV data to fit and optimize our classifier:
        clf_opt.fit(train_data_real, train_labels_real)
        # 4. retrieve optimal parameters:
        tmp = clf_opt.steps[-1][1]
        best_penalty = tmp.best_params_['penalty']
        # 5. Use the optimized classifier on the test dataset (w/o time):
        score = clf_opt.score(test_data_real, test_labels_real)

        print(score)
        print('best penalty: ' + best_penalty)
        # save the best params for later use
        file = open(bestParametersFile, "w")
        file.writelines('best penalty: ' + best_penalty)
        file.close()

    else:
        [best_penalty] = getBestParams(bestParametersFile)

    clf_tp = []
    CV_score = np.zeros((repeats * nFolds, tlim))
    Test_score_real = np.zeros((tlim))
    Test_score_omissions = np.zeros((tlim))
    clf_list = []

    for tp in np.arange(tlim):
        d2t_cv = train_data_real[:, :, tp]  # data to test
        d2t_test = test_data_real[:, :, tp]  # data to test - real

        clf = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(penalty=best_penalty))  # check the doc for multiclass
        # get CV score:
        CV_score[:, tp] = cross_val_score(clf, d2t_cv, train_labels_real, cv=rkf)

        # fit the model using all CV data:
        clf.fit(d2t_cv, train_labels_real)
        clf_list.append(clf)
        # generalize performance on test data:
        labels_test_estim = clf.predict(d2t_test)

        # le = LabelEncoder()
        test_labels_real1 = preprocessing.label_binarize(test_labels_real, classes=[1, 2, 3, 0])
        labels_test_estim1= preprocessing.label_binarize(labels_test_estim, classes=[1, 2, 3, 0])

        # test_labels_real = le.fit_transform(test_labels_real)
        # labels_test_estim = le.fit_transform(labels_test_estim)

        Test_score_real[tp] = roc_auc_score(test_labels_real1, labels_test_estim1, average='macro', multi_class='ovo')


        d2t_test_omissions = test_data[:, :, tp]  # data to test - omissions_corr lowConf
        labels_test_estim_omissions = clf.predict(d2t_test_omissions)

        labels_test_estim_omissions = preprocessing.label_binarize(labels_test_estim_omissions, classes=[1, 2, 3, 0])
        test_labels = preprocessing.label_binarize(test_labels, classes=[1, 2, 3, 0])
        Test_score_omissions[tp] = roc_auc_score(test_labels, labels_test_estim_omissions, average='macro', multi_class='ovo')


    results = [Test_score_real, Test_score_omissions, CV_score]
    np.save(outputfilename, results)

    with open(modelsFile, "wb") as f:
        for model in clf_list:
            pickle.dump(model, f)

    return results


def plot_MVPA(results, times, tlim, plotname, fullTrial=True, isBehavior=True):
    Test_score_real = results[0]
    CV_score = results[-1]
    if len(results) == 3:
        Test_score_omissions = results[1]
    elif len(results) == 4:
        Test_score_omissions = [results[1], results[2]]
    elif len(results) == 5:
        Test_score_omissions = [results[1], results[2], results[3]]
    # [Test_score_real, Test_score_omissions, CV_score] = results
    if not fullTrial:
        end_of_omission = np.where(times == 0.3)[0][0]
        times_omi = times[:end_of_omission + 1]
    else:
        times_omi = times
        end_of_omission = len(times) - 1

    fig = plt.figure(num=None, figsize=(8, 2), dpi=150)
    plt.subplot(1, 2, 1)
    ax = plt.plot(times_omi, Test_score_real[:end_of_omission + 1], label='Test Real')

    if len(results) == 4:
        if isBehavior:
            labels = ['Test Omissions_correct', 'Test Omissions_incorrect']
        else:
            labels = ['Test Omissions_lowConf', 'Test Omissions_highConf']

        ax = plt.plot(times_omi, Test_score_omissions[0][:end_of_omission + 1], label=labels[0])
        ax = plt.plot(times_omi, Test_score_omissions[1][:end_of_omission + 1], label=labels[1])

    elif len(results) == 5:
        labels = ['Test Omissions - 80%', 'Test Omissions - 90%', 'Test Omissions - 100%']
        for i in range(len(Test_score_omissions)):
            ax = plt.plot(times_omi, Test_score_omissions[i][:end_of_omission + 1], label=labels[i])

    else:
        ax = plt.plot(times_omi, Test_score_omissions[:tlim], label='Test Omissions')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

    plt.title('Test set')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    ax = plt.plot(times, np.nanmean(CV_score, axis=0)[:tlim], label='CV Real')
    plt.title('Cross-validation set')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3))

    fig1 = plt.gcf()
    fig1.savefig(plotname, bbox_inches='tight')
    plt.show()


def plot_MVPA_Group(results, labels, plotname):
    print(len(results))
    nrow = int(len(results) / 2) - 1
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=150)

    print(axs.shape)
    # print(len(axs[0]))
    if nrow > 1:
        for i in range(nrow):

            if i == 0:
                axs[i, 0].plot(epochs.times[:tlim], np.nanmean(results[0], axis=0)[:tlim], label=labels[0])
                print(results[i][:tlim])
            else:
                axs[i, 0].plot(epochs.times[:tlim], results[i][:tlim], label=labels[i])
                axs[i, 1].plot(epochs.times[:tlim], results[i * loop_size + 2][:tlim], label=labels[i * loop_size + 2])
                axs[i, 1].plot(epochs.times[:tlim], results[i * loop_size + 3][:tlim], label=labels[i * loop_size + 3])

    elif nrow == 1:
        axs[0].plot(epochs.times[:tlim], np.nanmean(results[0], axis=0)[:tlim], label=labels[0])
        axs[1].plot(epochs.times[:tlim], results[1][:tlim], label=labels[1])
        axs[1].plot(epochs.times[:tlim], results[2][:tlim], label=labels[2])
        axs[1].plot(epochs.times[:tlim], results[3], label=labels[3])

    for i in range(len(axs.flat)):
        if i == 0:
            axs.flat[i].set_title('Cross-validation set')
        else:
            axs.flat[i].set_title('Test set')
        axs.flat[i].set(xlabel='Time (s)', ylabel='Accuracy')
        axs.flat[i].legend(loc='upper center', bbox_to_anchor=(0.1, -0.3))

    # plt.xlabel('Time (s)')

    fig.tight_layout()
    fig1 = plt.gcf()
    fig1.savefig(plotname, bbox_inches='tight')
    plt.show()


def concatNonEmpty(lists):
    newList = []
    for l in lists:
        if len(l) > 0:
            if len(newList) > 0:
                newList = np.concatenate((newList, l))
            else:
                newList = l
    return newList


def plot_conditions(data, times, ylabel, plotname, labels=None):
    sns.set(style="white")
    ColorsL = np.array(([228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0])) / 256
    col_axes = np.array((82, 82, 82)) / 256

    al = 0.2
    fig = plt.figure(num=None, figsize=(4, 2), dpi=150)

    epochs_mean = np.mean(data, axis=0)
    # print(epochs_mean.shape)
    # print(times.shape)
    epochs_std = sem(data, axis=0) / 2
    # print(epochs_std)

    plt.plot(times, epochs_mean, color=ColorsL[0], linewidth=2, label=labels)
    plt.fill_between(times, epochs_mean, epochs_mean + epochs_std, color=ColorsL[0], interpolate=True, alpha=al)
    plt.fill_between(times, epochs_mean, epochs_mean - epochs_std, color=ColorsL[0], interpolate=True, alpha=al)
    plt.ylabel(ylabel)
    plt.xlabel('Times')
    plt.savefig(plotname, bbox_inches='tight')


def averageCVScores(CV_score_time):
    avg_cv_scores = []
    for cv in CV_score_time:
        avg_scores_tmp = []
        sum_col_wise = cv.sum(axis=0)
        avg_scores_tmp = [s/len(cv) for s in sum_col_wise]
        avg_cv_scores.append(avg_scores_tmp)
    return avg_cv_scores


# Load confidence data
def loadConfData(confFile):
    conf = np.load(confFile)
    # Convert confidence values to int and None to -1 to ease their use
    for i in range(len(conf)):
        if conf[i] is not None:
            if len(conf[i]) > 0:
                conf[i] = int(conf[i][0])
            else:
                conf[i] = 0
        else:
            conf[i] = -1
    return conf


def splitEpochs_byConfidence(confFile, epochs):
    conf = loadConfData(confFile)

    # print('number of epochOmissionAR: ', len(epochOmissionAR))
    # print('number of confidence values: ', len(conf))

    # Extract the unique confidence values which are not None in data
    conf_values_unique = np.unique([c for c in conf if c > 0])
    # print("Unique confidence values: ", conf_values_unique)

    conf = np.array(conf)
    conf_low_indices = np.where((conf <= conf_values_unique[1]) & (conf > 0))[0]

    print("Number of low confidence responses: ", len(conf_low_indices))
    conf_high_indices = np.where(conf >= conf_values_unique[2])[0]

    print("Number of high confidence responses: ", len(conf_high_indices))
    print('Number of None ( = -1): ', len(np.where(conf == -1)[0]))
    print('Number of no-resp ( = 0): ', len(np.where(conf == 0)[0]))
    print('Total confidence questions: ', len(np.where(conf > -1)[0]))

    low_conf_epochs = epochs[conf_low_indices]
    print(len(low_conf_epochs))
    high_conf_epochs = epochs[conf_high_indices]
    print(len(high_conf_epochs))

    return low_conf_epochs, high_conf_epochs


def prepareData_conf_behavior(label, epochs):
    if label in epochs.event_id:
        new_epochs = epochs[label]
        data = new_epochs.get_data()
    else:
        data = []
    # print('num data len: ', len(data))

    return data


def prepareData_conf_pred(data_omission_living_lowConf_list, data_omission_living_highConf_list,
                          data_omission_obj_lowConf_list, data_omission_obj_highConf_list):
    # Low Confidence
    data_omission_living_lowConf = concatNonEmpty(data_omission_living_lowConf_list)
    labels_omission_living_lowConf = np.zeros(len(data_omission_living_lowConf))
    # print('living low conf: ', len(labels_omission_living_lowConf))

    data_omission_obj_lowConf = concatNonEmpty(data_omission_obj_lowConf_list)
    labels_omission_obj_lowConf = np.ones(len(data_omission_obj_lowConf))
    # print('obj low conf: ', len(labels_omission_obj_lowConf))

    # Combine
    data_omission_lowConf = concatNonEmpty([data_omission_living_lowConf, data_omission_obj_lowConf])
    labels_omission_lowConf = concatNonEmpty([labels_omission_living_lowConf, labels_omission_obj_lowConf])

    # High Confidence

    data_omission_living_highConf = concatNonEmpty(data_omission_living_highConf_list)
    labels_omission_living_highConf = np.zeros(len(data_omission_living_highConf))
    # print('living high conf: ', len(labels_omission_living_highConf))

    data_omission_obj_highConf = concatNonEmpty(data_omission_obj_highConf_list)
    labels_omission_obj_highConf = np.ones(len(data_omission_obj_highConf))
    # print('obj high conf: ', len(labels_omission_obj_highConf))

    # Combine
    data_omission_highConf = concatNonEmpty([data_omission_living_highConf, data_omission_obj_highConf])
    labels_omission_highConf = concatNonEmpty([labels_omission_living_highConf, labels_omission_obj_highConf])

    return data_omission_lowConf, data_omission_highConf, labels_omission_lowConf, labels_omission_highConf


def prepareData_pred_behavior_conf(living_data_list, obj_data_list):
    data_living_all = concatNonEmpty(living_data_list)
    data_obj_all = concatNonEmpty(obj_data_list)

    labels_living_all = np.zeros(len(data_living_all))
    labels_obj_all = np.ones(len(data_obj_all))

    data_all = concatNonEmpty([data_living_all, data_obj_all])
    labels_all = concatNonEmpty([labels_living_all, labels_obj_all])

    return data_all, labels_all


def simple_line_plot(y):
    ax = plt.axes()
    ax.plot(np.arange(1, len(y)+1), y)
    # plt.ylim(0, 1)
    plt.xlim(0, 180)
    plt.xlabel('timepoints')
    plt.ylabel('ROC_AUC score')
    plt.grid(True)
    plt.title("Performance on Train data")
    plt.show()


def plot_base_curve(y1, y2, y3):
    ax = plt.axes()
    x = np.arange(-0.5, 1.3, 0.01)

    ax.plot(x, y1, color='b', label='Train')
    ax.plot(x, y2, color='r', label='Test')
    ax.plot(x, y3, color='g', label='CV')

    plt.ylim(0.13, 0.76)
    plt.xlim(-0.5, 1.3)
    plt.xlabel('time (secs)')
    plt.ylabel('accuracy')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1.0),bbox_transform=plt.gcf().transFigure)
    plt.legend(loc='upper right')
    plt.title(f"Classification Performance over time")
    df = 51
    plt.savefig(path.join(r"\MEG Plots", str("%02d" % df)))
    plt.close("all")


def plot_smoothed(df, y):

    # Generate spline basis with different degrees of freedom
    x_basis = cr(x, df=df, constraints="center")
    # Fit model to the data
    model = LinearRegression().fit(x_basis, y)

    # Get estimates
    y_hat = model.predict(x_basis)

    plt.plot(x, y_hat1, color='b', label='Train')
    plt.plot(x, y_hat2, color='r', label='Test')
    plt.plot(x, y_hat3, color='g', label='CV')

    plt.ylim(0.13, 0.76)
    plt.xlim(-0.5, 1.3)
    plt.xlabel('time (secs)')
    plt.ylabel('accuracy')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1.0),bbox_transform=plt.gcf().transFigure)
    plt.legend(loc='upper right')
    plt.title(f"Classification Performance over time")
    plt.savefig(path.join(r"\MEG Plots", str("%02d" % df)))
    plt.close("all")


def plot_smooth_curve(df, y1, y2, y3):

    x = np.arange(-0.5, 1.3, 0.01)
    # Generate spline basis with different degrees of freedom
    x_basis = cr(x, df=df, constraints="center")
    # Fit model to the data
    model1 = LinearRegression().fit(x_basis, y1)
    model2 = LinearRegression().fit(x_basis, y2)
    model3 = LinearRegression().fit(x_basis, y3)


    # Get estimates
    y_hat1 = model1.predict(x_basis)
    y_hat2 = model2.predict(x_basis)
    y_hat3 = model3.predict(x_basis)

    ax = plt.axes()
    # plt.plot(x, y_hat, label=f"df={df}")

    ax.plot(x, y_hat1, color='b', label='Train')
    ax.plot(x, y_hat2, color='r', label='Test')
    ax.plot(x, y_hat3, color='g', label='CV')

    plt.ylim(0.13, 0.76)
    plt.xlim(-0.5, 1.3)
    plt.xlabel('time (secs)')
    plt.ylabel('accuracy')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1.0),bbox_transform=plt.gcf().transFigure)
    plt.legend(loc='upper right')
    plt.title(f"Classification Performance over time")
    plt.savefig(path.join(r"\MEG Plots", str("%02d" % df)))
    plt.close("all")


def plot_smoothed_2curves_inter(df, y1, y2, y3):
    x = np.arange(0, len(y1) / 100, 0.01)

    box = np.ones(df) / df
    y1_smooth = np.convolve(y1, box, mode='same')
    y2_smooth = np.convolve(y2, box, mode='same')
    y3_smooth = np.convolve(y3, box, mode='same')

    ax = plt.axes()
    # plt.plot(x, y_hat, label=f"df={df}")

    ax.plot(x, y1_smooth, color='b', label='Train')
    ax.plot(x, y2_smooth, color='r', label='Test')
    ax.plot(x, y3_smooth, color='g', label='CV')

    plt.ylim(0.2, 0.9)
    plt.xlim(0, 1.8)
    plt.xlabel('time (secs)')
    plt.ylabel('ROC_AUC score')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1.0),bbox_transform=plt.gcf().transFigure)
    plt.legend(loc='upper right')
    plt.title(f"Classification Performance over time")
    plt.savefig(path.join(r"\MEG Plots", f"conv"+str("%02d" % df)))
    # plt.show()
    plt.close("all")