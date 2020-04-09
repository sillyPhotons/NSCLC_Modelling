"""
    This file contains utility functions which can be used read comma separated 
    data into numpy arrays, and plot visiualize them
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl


def func(x, a):
    """
    An exponential function used to fit the no-treatment data found in 
    Detterbeck 2008.

    x: numpy array of a scalar
    a: decay parameter 
    """
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


def get_fitted_data(file_path, increment, interval = [0, 120]):
    """
    Returns the x,y array of data at `file_path` that has been fitted with an exponential decay model, with f(0) = 1

    Requires: file at file_path exists 

    `file_path`: string, path to the data file
    `increment`: difference between x values in the array returned
    `interval`: unused
    """
    dat = np.loadtxt(file_path, delimiter=',')
    x, y = read_file(dat)
    popt, pcov = curve_fit(func, x, y)
    fitted_x = np.arange(interval[0], interval[1] + increment, increment)
    fitted_y = func(fitted_x, popt[0])

    return fitted_x, fitted_y


def plot(data_array, curve_label, xlabel, ylabel, title, fit = False):
    """
    `data_array`: numpy array whose elements are order pairs (x,y)
    `curve_label`: name of the curve as seen in the legend
    `x_label`: label to the x axis
    `y_label`: label to the y axis
    `title`: title to the plot
    `fit`: if True, a fitted exponential decay curve will be plotted as well
    """
    x, y = read_file(data_array)
    popt, pcov = curve_fit(func, x, y)
    fitted_x = np.arange(min(x), max(x) + 0.5, 0.5)
    fitted_y = func(fitted_x, *popt)

    if fit:
        plt.plot(fitted_x, fitted_y, label="Fitted Curve")
    
    plt.step(x, y, label=curve_label, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.minorticks_on()
    plt.title(title)


if __name__ == "__main__":

    mpl.rcParams["font.family"] = "FreeSerif"
    plt.rc("text", usetex=True)
    plt.figure(dpi=100)

    s1 = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
    s2 = np.loadtxt("./Data/stage2Better.csv", delimiter=',')
    s3A = np.loadtxt("./Data/stage3ABetter.csv", delimiter=',')
    s3B = np.loadtxt("./Data/stage3BBetter.csv", delimiter=',')
    s4 = np.loadtxt("./Data/stage4Better.csv", delimiter=',')

    # plot and show the data
    plot(s1, "Stage 1", "Survival Time [Months]",
         "Proportion of Patients Alive", "KMSC From Data")
    plot(s2, "Stage 2", "Survival Time [Months]",
         "Proportion of Patients Alive", "KMSC From Data")
    plot(s3A, "Stage 3A", "Survival Time [Months]",
         "Proportion of Patients Alive", "KMSC From Data")
    plot(s3B, "Stage 3B", "Survival Time [Months]",
         "Proportion of Patients Alive", "KMSC From Data")
    plot(s4, "Stage 4", "Survival Time [Months]",
         "Proportion of Patients Alive", "KMSC From Data")
    plt.show()
    
