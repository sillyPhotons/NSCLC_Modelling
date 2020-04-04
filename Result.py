from lmfit import fit_report
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

mpl.rcParams["font.family"] = "FreeSerif"
plt.rc("text", usetex=True)
plt.figure(dpi=100)


class ResultObj():

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


class ResultManager():

    def __init__(self, primary_path="."):
        now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        self.directory_path = primary_path + "/Results/sim_{}".format(now)
        os.mkdir(self.directory_path)

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
        plt.savefig(path + "/plot.pdf")