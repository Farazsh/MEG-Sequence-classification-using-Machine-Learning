from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from patsy.mgcv_cubic_splines import cr
from scipy.stats import sem
from sklearn.linear_model import LinearRegression


def annotate_max(xmax, ymax, ax=None, xytext=(0.95, 0.85), ha='right', angleA=0, angleB=60):
    """
    Annotate the maximum point on the plot.

    Parameters:
        xmax (float): x-coordinate of the maximum point.
        ymax (float): y-coordinate of the maximum point.
        ax (matplotlib.axes.Axes, optional): Axes object to annotate.
        xytext (tuple, optional): Text position in axes fraction coordinates.
        ha (str, optional): Horizontal alignment of the text.
        angleA (float, optional): Starting angle of the arrow.
        angleB (float, optional): Ending angle of the arrow.
    """
    if ax is None:
        ax = plt.gca()

    text = f"Time: {xmax}ms, acc={ymax:.4f}"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle=f"angle,angleA={angleA},angleB={angleB}")
    annotation_kwargs = dict(
        xycoords='data',
        textcoords="axes fraction",
        arrowprops=arrowprops,
        bbox=bbox_props,
        ha=ha,
        va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=xytext, **annotation_kwargs)


def plot_smooth_curve(df, train_acc, test_acc, cv_acc_list, save_path=None):
    """
    Plots the smooth accuracy curves from numpy arrays using cubic splines.

    Parameters:
        df (int): Degrees of freedom for the spline basis.
        train_acc (array-like): Training accuracy data.
        test_acc (array-like): Test accuracy data.
        cv_acc_list (array-like): List of cross-validation accuracy data arrays.
        save_path (str, optional): Path to save the plot image. If None, the plot is displayed.
    """
    # Define time points
    time_points = np.arange(-500, 1300, 10)

    # Generate spline basis with specified degrees of freedom
    spline_basis = cr(time_points, df=df, constraints="center")

    # Fit linear regression models to the data
    model_train = LinearRegression().fit(spline_basis, train_acc)
    model_test = LinearRegression().fit(spline_basis, test_acc)

    # Get predictions
    y_hat_train = model_train.predict(spline_basis)
    y_hat_test = model_test.predict(spline_basis)

    # Process cross-validation data
    y_hat_cv_list = []
    for cv_acc in cv_acc_list:
        model_cv = LinearRegression().fit(spline_basis, cv_acc)
        y_hat_cv = model_cv.predict(spline_basis)
        y_hat_cv_list.append(y_hat_cv)
    y_hat_cv_array = np.array(y_hat_cv_list)
    y_hat_cv_mean = y_hat_cv_array.mean(axis=0)

    # Reference line at chance level
    chance_level = 0.25
    y_ref = np.full_like(time_points, chance_level)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(time_points, y_ref, '--', color='#b2abd2', label='Theoretical chance level')
    ax.plot(time_points, y_hat_test, color='#5e3c99', label='Test')

    # Optionally plot other curves (currently commented out)
    # ax.plot(time_points, y_hat_train, color='r', label='Train')
    # ax.plot(time_points, y_hat_cv_mean, color='#B7950B', label='CV')

    # Setting plot limits and labels
    ax.set_ylim(0.15, 0.6)
    ax.set_xlim(-500, 1300)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy')

    # Customize legend order
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left',
        fontsize='small',
    )

    ax.set_title("Classification Performance over Time")

    # Annotate maximum accuracy
    xmax = time_points[np.argmax(y_hat_test)]
    ymax = y_hat_test.max()
    annotate_max(xmax, ymax, ax)

    # Annotate maximum accuracy in later time window
    later_indices = 90  # Adjust as needed
    _time_points = time_points[later_indices:]
    _y_hat_test = y_hat_test[later_indices:]
    xmax_late = _time_points[np.argmax(_y_hat_test)]
    ymax_late = _y_hat_test.max()
    annotate_max(
        xmax_late,
        ymax_late,
        ax,
        xytext=(0.98, 0.65),
        ha='right',
        angleA=0,
        angleB=60,
    )

    # Show or save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close('all')


def plot_base_curve(train_acc, test_acc, cv_acc_list, save_path=None):
    """
    Plots the base accuracy curves without smoothing.

    Parameters:
        train_acc (array-like): Training accuracy data.
        test_acc (array-like): Test accuracy data.
        cv_acc_list (array-like): List of cross-validation accuracy data arrays.
        save_path (str, optional): Path to save the plot image. If None, the plot is displayed.
    """
    # Define time points
    time_points = np.arange(-500, 1300, 10)

    # Calculate mean of cross-validation accuracies
    cv_acc_mean = cv_acc_list.mean(axis=0)

    # Reference line at chance level
    chance_level = 0.25
    y_ref = np.full_like(time_points, chance_level)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(time_points, train_acc, color='r', label='Train')
    ax.plot(time_points, test_acc, color='#b2abd2', label='Test')
    ax.plot(time_points, cv_acc_mean, color='#fdb863', label='CV')
    ax.plot(time_points, y_ref, color='#5e3c99', label='Theoretical chance level')

    # Plot standard error around cross-validation curve
    alpha = 0.3
    std_error = sem(cv_acc_list, axis=0) / 2
    ax.fill_between(
        time_points,
        cv_acc_mean + std_error,
        cv_acc_mean - std_error,
        color='y',
        alpha=alpha,
        label='Std dev',
    )

    # Set plot limits and labels
    ax.set_ylim(0.13, 0.88)
    ax.set_xlim(-500, 1300)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper left', fontsize='small')
    ax.set_title("Classification Performance over Time")

    # Annotate maximum accuracy
    xmax = time_points[np.argmax(train_acc)]
    ymax = train_acc.max()
    annotate_max(xmax, ymax, ax)

    # Annotate maximum accuracy in later time window
    later_indices = 90
    _time_points = time_points[later_indices:]
    _train_acc = train_acc[later_indices:]
    xmax_late = _time_points[np.argmax(_train_acc)]
    ymax_late = _train_acc.max()
    annotate_max(
        xmax_late,
        ymax_late,
        ax,
        xytext=(0.98, 0.65),
        ha='right',
        angleA=0,
        angleB=60,
    )

    # Show or save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close('all')


def main():
    # Define results directory and file
    results_dir = 'path_to_results_directory'  # Update this path
    results_name = 'MEGAnalysisResult'
    result_file = join(results_dir, results_name + ".npy")

    # Load results
    results = np.load(result_file, allow_pickle=True)
    train_acc = results[0]
    test_acc = results[1]
    cv_acc_list = results[2]

    # Plot base curve
    # plot_base_curve(train_acc, test_acc, cv_acc_list, save_path='base_curve.png')

    # Plot smooth curve with specified degrees of freedom
    df = 46
    plot_smooth_curve(df, train_acc, test_acc, cv_acc_list, save_path='smooth_curve.png')


if __name__ == '__main__':
    main()
