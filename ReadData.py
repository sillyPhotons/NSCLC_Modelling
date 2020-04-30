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
    Exponential decay function, returns the value of e^(-a*x) given arugments 
    `x` and `a`. If `x` is a numpy array, then a numpy array is returned, if
    `x` is scalar, then a scalar is returned

    Params::
        `x`: `numpy` array of a scalar
        
        `a`: a scalar decay parameter 
    """
    return np.exp(-a*x)

def read_file(file_path, interval = [0, 60]):
    """
    Returns two numpy arrays respectively representing the x and y series in a
    *.csv file specified by the `file_path` parameter. The returned numpy arrays
    is the intersection between `interval` and the original domain of the x
    data in the data file  

    Params::
        `file_path`: string to a *.csv data file with 2 columns. Each row represents one ordered (x,y) pair.
        
        `interval`: a python list, or a numpy array with 2 elements. `interval[0]` represents the lower bound and `interval[1]` represents the upper bound of the domain
    
    Requires:
        `interval[0] >= 0`
        `interval[1] >= 0`
        `interval[1] >= interval[0]`

    Raises:
        Assertion errors for violations of requirements
    """

    assert(interval[0] >= 0 and interval[1] >= 0)
    assert(interval[0] <= interval[1])

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

        x = np.concatenate(x, axis=0 )
        y = np.concatenate(y, axis=0 )

    else:
        x = np.array([])
        y = np.array([])
    
    return x,y


def get_fitted_data(x, y, increment):
    """
    Given `x` and `y` numpy arrays, returns two arrays representing points on a 
    KMSC that has been fitted with an exponential model. The x increments of 
    that data points are given by the scalar value `increment`.

    Params::
        `x`: numpy array

        `y`: numpy array

        `increment`: scalar value

    Requires::
        `increment > 0`
        `x.size > 0`
        `y.size > 0`

    Raises::
        Assertion error if `increment <= 0`
    """

    assert(increment > 0)
    assert(x.size > 0 and y.size > 0)
    
    popt, pcov = curve_fit(exponential, x, y)
    fitted_x = np.arange(x[0], x[-1] + increment, increment)
    fitted_y = exponential(fitted_x, popt[0])

    return fitted_x, fitted_y

def plot(x, y, curve_label, xlabel, ylabel, title, fit = False):
    """
    Plots the given `x` and `y` array with the given `xlabel`, `ylabel`, 
    `title`. If `fit = True`, then an exponential fit is also drawn on the same 
    plot

    Params::
        `x`: numpy array

        `y`: numpy array

        `curve_label`: string, label of the curve as seen in the legend of generated plot

        `xlabel`: string, label of the x axis

        `ylabel`: string, label of the y axis

        `title`: string, title of the plot

        `fit`: boolean
    """
    popt, pcov = curve_fit(exponential, x, y)
    fitted_x = np.arange(x[0], x[-1] + 0.5, 0.5)
    fitted_y = exponential(fitted_x, *popt)

    if fit:
        plt.plot(fitted_x, fitted_y, label="Fitted Curve")
    
    plt.scatter(x, y, label=curve_label, alpha=0.7, s = 25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.minorticks_on()
    plt.title(title)
    # plt.show()

# import Constants as c
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rc("text", usetex=True)
# # plt.rcParams['font.family'] = 'serif'
# plt.rcParams.update({'font.size': 18,
#                      'figure.autolayout': True})
# for key in c.TABLE2.keys():
#     x,y = read_file("./Data/stage{}Better.csv".format(key), [0,59])
#     plot(x, y, "Stage {}".format(key), "Time [months]", "Proportion of Patients Alive", "")

# plt.savefig("untreated_data.pdf")