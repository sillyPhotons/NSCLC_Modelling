"""
Author: Ruiheng Su 2020
    
This file contains utility functions used to parse csv files into numpy arrays, 
and visiualize the data
"""
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential(x, a):
    """
    Exponential decay function, returns the value of e^(-a*x) given arugments `x` and `a`. If `x` is a numpy array, then a numpy array is returned, if `x` is scalar, then a scalar is returned

    `x`: `numpy` array of a scalar
    `a`: a scalar decay parameter 
    """
    return np.exp(-a*x)

def read_file(file_path, interval = [0, 60]):
    """
    
    """

    assert(interval[0] >= 0 and interval[1] >= 0)
    dat = np.loadtxt(file_path, delimiter=',')
    x,y = np.hsplit(dat, 2)
    x = np.around(x)

    if interval[0] < x[0][0] or interval[1] > x[-1][0]:
        logging.warning("Interval requested {} is not a subset of the data domain {}.".format(interval, [x[0][0], x[-1][0]]))

    lb = np.where(x >= interval[0])[0]
    ub = np.where(x <= interval[1])[0]
   

    if lb.size != 0 and ub.size != 0:
        x = x[lb[0]:ub[-1]+1]
        y = y[lb[0]:ub[-1]+1]
    elif lb.size != 0 and ub.size == 0:
        x = x[lb[0]:]
        y = y[lb[0]:]
    elif lb.size == 0 and ub.size != 0:
        x = x[:ub[-1]]
        y = y[:ub[-1]]

    x = np.concatenate(x, axis=0 )
    y = np.concatenate(y, axis=0 )

    assert(x.size == y.size)

    return x,y


def get_fitted_data(x, y, increment):
    """
    
    """
    popt, pcov = curve_fit(exponential, x, y)
    fitted_x = np.arange(x[0], x[-1] + increment, increment)
    fitted_y = exponential(fitted_x, popt[0])

    return fitted_x, fitted_y


def plot(x, y, curve_label, xlabel, ylabel, title, fit = False):
    """
    
    """
    popt, pcov = curve_fit(exponential, x, y)
    fitted_x = np.arange(x[0], x[-1] + 0.5, 0.5)
    fitted_y = exponential(fitted_x, *popt)

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
    
    print(read_file("./Data/stage1Better.csv", interval=[20, 30]))
    # s1 = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
    # s2 = np.loadtxt("./Data/stage2Better.csv", delimiter=',')
    # s3A = np.loadtxt("./Data/stage3ABetter.csv", delimiter=',')
    # s3B = np.loadtxt("./Data/stage3BBetter.csv", delimiter=',')
    # s4 = np.loadtxt("./Data/stage4Better.csv", delimiter=',')
    # radiation = np.loadtxt("./Data/radiotherapy.csv", delimiter=',')

    # # plot and show the data
    # # plot(s1, "Stage 1", "Survival Time [Months]",
    # #      "Proportion of Patients Alive", "KMSC From Data")
    # # plot(s2, "Stage 2", "Survival Time [Months]",
    # #      "Proportion of Patients Alive", "KMSC From Data")
    # # plot(s3A, "Stage 3A", "Survival Time [Months]",
    # #      "Proportion of Patients Alive", "KMSC From Data")
    # # plot(s3B, "Stage 3B", "Survival Time [Months]",
    # #      "Proportion of Patients Alive", "KMSC From Data")
    # # plot(s4, "Stage 4", "Survival Time [Months]",
    # #      "Proportion of Patients Alive", "KMSC From Data")


    # plot(radiation, "Radiotherapy", "Survival Time [Months]", "Proportion of Patients Alive", "KMSC From Data")    
    # plt.show()
    
