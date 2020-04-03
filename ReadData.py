import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl


def func(x, a):
    return np.exp(-a*x)


def read_file(data_array):
    x = []
    y = []

    for num in range(len(data_array)):

        x.append(data_array[num][0])
        y.append(data_array[num][1])

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    return x, y


"""
    Returns estimated experimental data in a 120 month period. Increment is the
    number of months between each data point returned 

    Requires: file at file_path exists 
"""


def get_data(file_path, increment, range=[0, 120]):

    dat = np.loadtxt(file_path, delimiter=',')
    x, y = read_file(dat)
    popt, pcov = curve_fit(func, x, y)
    fitted_x = np.arange(range[0], range[1] + increment, increment)
    fitted_y = func(fitted_x, popt[0])

    return fitted_x, fitted_y


def plot(data_array, curve_label, xlabel, ylabel, title):

    x, y = read_file(data_array)
    popt, pcov = curve_fit(func, x, y)
    fitted_x = np.arange(min(x), max(x) + 0.5, 0.5)
    fitted_y = func(fitted_x, *popt)

    plt.plot(fitted_x, fitted_y, label="Fitted Curve")
    plt.scatter(x, y, label=curve_label, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.minorticks_on()
    plt.title(title)


if __name__ == "__main__":

    mpl.rcParams["font.family"] = "FreeSerif"
    plt.rc("text", usetex=True)
    plt.figure(dpi=100)

    s1 = np.loadtxt("./Data/stage1.csv", delimiter=',')
    s2 = np.loadtxt("./Data/stage2.csv", delimiter=',')
    s3A = np.loadtxt("./Data/stage3A.csv", delimiter=',')
    s3B = np.loadtxt("./Data/stage3B.csv", delimiter=',')
    s4 = np.loadtxt("./Data/stage4.csv", delimiter=',')

    plot(s1, "Stage 1", "Survival Time [Months]",
         "Proportion of Patients Alive", "Survival Curve")
    plot(s2, "Stage 2", "Survival Time [Months]",
         "Proportion of Patients Alive", "Survival Curve")
    plot(s3A, "Stage 3A", "Survival Time [Months]",
         "Proportion of Patients Alive", "Survival Curve")
    plot(s3B, "Stage 3B", "Survival Time [Months]",
         "Proportion of Patients Alive", "Survival Curve")
    plot(s4, "Stage 4", "Survival Time [Months]",
         "Proportion of Patients Alive", "Survival Curve")
    plt.show()
