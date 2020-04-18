"""
Author: Ruiheng Su

File containing the defintion of two ulility classes to deal with
results of simulations
"""

import os
import csv
import matplotlib as mpl
from lmfit import fit_report
from datetime import datetime
import matplotlib.pyplot as plt


class ResultObj():
    """
    Represents the result of a minimization process or a predictive 
    model
    """

    def __init__(self, plt_callable, x, y, xdes="", ydes="", curve_label="", **plotargs):
        """
        Params::
            `plt_callable`: a callable to which takes `plotargs` as arugments. A function which generates a plot

            `x`: numpy array of the independent variable

            `y`: numpy array of the dependent varialbe

            `xdes`: string description of `x`, used for `xlabel` when plotting

            `ydes`: string description of `y`, used for `ylabel` when plotting

            `curve_label`: string label used both as the name of a data file generated and the label of the curve in the legend of the plot

            `**plotargs`: keyword arguments for `plt_callable`

        Raises::
            Assertion error if `len(x) != len(y)`
        """
        self.plotfunc = plt_callable
        self.data_size = len(x)
        self.x = x
        self.y = y
        self.xdes = xdes
        self.ydes = ydes
        self.curve_label = curve_label
        self.plotargs = plotargs

        assert(self.data_size == len(y))


class ResultManager():
    """
    Class of objects which records, plots, 
    timestamps, saves a result represented by a ResultObj.
    """

    def __init__(self, primary_path="."):
        """
        Params::
            `primary_path`: path to a folder named `Results` where results are stored. If there is no folder named "Results" in the `primary_path`, then a folder with the name `Results` is created. All simulation results will be stored in separate timestamped folders inside of the `Results` folder  
        """
        now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.directory_path = primary_path + "/Results/sim_{}".format(now)
        os.makedirs(self.directory_path)

    def record_simulation(self, result, *args, comment="stage_[N]"):
        """
        Given various `ResultObj` objects, stores and plots the data contained 
        within the `ResultObj` objects in a folder with the name as the
        `comment` string preceeding a timestamp

        Params::
            `result`: a `MinimizerResult` object

            `*args*`: a variable number of `ResultObj` objects

            `comment`: string which preceeds the folder name
        """

        plt.rc("text", usetex=True)
        plt.rcParams['font.family'] = 'serif'

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
        plt.close()

    def record_prediction(self, *args, comment="stage_[N]"):
        """
        Given various `ResultObj` objects, stores and plots the data contained 
        within the `ResultObj` objects in a folder with the name as the
        `comment` string preceeding a timestamp

        Params::
            `result`: a `MinimizerResult` object

            `*args*`: a variable number of `ResultObj` objects

            `comment`: string which preceeds the folder name
        """

        plt.rc("text", usetex=True)
        plt.rcParams['font.family'] = 'serif'

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
        plt.savefig(path + "/plot.pdf")
        plt.close()
