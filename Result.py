"""
    This file contains the defintion of two ulility classes to deal with
    results of simulations
"""

from lmfit import fit_report
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

plt.rc("text", usetex=True)
plt.rcParams['font.family'] = 'serif'


"""
    ResultObj represents the result of a minimization process, or the evaluation
    of a predictive model
"""
class ResultObj():

    """
        plt_callable: a callable to which plotargs can be arugment of
        x: numpy array of the independent variable
        y: numpy array of the dependent varialbe
        xdes: string description of `x`, used for xlabel when plotting
        ydes: string description of `y`, used for ylable when plotting
        curve_label = string label used both as the name of a data file 
        generated and the label of the curve in the legend of the plot
        **plotargs: keyword arguments for `plt_callable`
    
    """
    def __init__(self, plt_callable, x, y, xdes="", ydes="", curve_label="", **plotargs):
        self.plotfunc = plt_callable
        self.data_size = len(x)
        self.x = x
        self.y = y
        self.xdes = xdes
        self.ydes = ydes
        self.curve_label = curve_label
        self.plotargs = plotargs

        assert(self.data_size == len(y))


"""
    ResultManager class representing result managers which records, plots, 
    timestamps, saves a result represented by a ResultObj.
"""
class ResultManager():

    """
        primary_path: path to a file where results are stored. If there is no
        folder named "Results" in the `primary_path`, then such a folder is 
        created. All simulation results will be stored in a timestamp 
        subdirectory of the `Results` directory  
    """
    def __init__(self, primary_path="."):
        now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.directory_path = primary_path + "/Results/sim_{}".format(now)
        os.makedirs(self.directory_path)

    """
        Given a `MinimizerResult` object, and various `ResultObj` objects, 
        stores and plots the data contained within the `ResultObj` objects in a 
        timestamped directory with its name appended `comments`, in a folder
        "Results" folder. 
    """
    def record_simulation(self, result, *args, comment="stage_[N]"):

        now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        path = self.directory_path + "/{}_{}".format(comment, now)
        os.mkdir(path)

        with open(path + "/report.txt", "w") as report:
            report.write(fit_report(result))

        for r in args:
            description = r.curve_label
            with open(path + "/{}.csv".format(description), "w") as datafile:

                writer = csv.writer(datafile, delimiter=",")

                for num in range(r.data_size):
                    writer.writerow([r.x[num], r.y[num]])

            r.plotfunc(r.x, r.y, **r.plotargs)
            plt.xlabel("Months")
            plt.ylabel("Proportion of Patients Alive")
        plt.legend()
        plt.savefig(path + "/plot.pdf")

    """
        Given various `ResultObj` objects, stores and plots the data contained 
        within the `ResultObj` objects in a timestamped directory with its name
        appended `comments`, in a folder "Results" folder. 
    """

    def record_prediction(self, *args, comment="stage_[N]"):

        now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        path = self.directory_path + "/{}_{}".format(comment, now)
        os.mkdir(path)

        for r in args:
            description = r.curve_label
            with open(path + "/{}.csv".format(description), "w") as datafile:

                writer = csv.writer(datafile, delimiter=",")

                for num in range(r.data_size):
                    writer.writerow([r.x[num], r.y[num]])

            r.plotfunc(r.x, r.y, **r.plotargs)
            plt.xlabel("Months")
            plt.ylabel("Proportion of Patients Alive")
        plt.legend()
        # plt.show()
        plt.savefig(path + "/plot.pdf")
        # plt.close()