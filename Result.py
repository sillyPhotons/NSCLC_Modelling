from lmfit import fit_report
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv


class ResultObj():

    def __init__(self, x, y, xdes="", ydes="", curve_label="", **plotargs):
        self.data_size = len(x)
        self.x = x
        self.y = y
        self.xdes = xdes
        self.ydes = ydes
        self.curve_label = curve_label
        self.plotargs = plotargs

        assert(self.data_size == len(y))


def record_simulation(result, *args, stage = "stage_[N]"):

    mpl.rcParams["font.family"] = "FreeSerif"
    plt.rc("text", usetex=True)
    plt.figure(dpi=100)

    now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    directory_path = "./Results/sim_{}_{}".format(stage, now)
    os.mkdir(directory_path)

    with open(directory_path + "/report.txt", "w") as report:
        report.write(fit_report(result))

    for r in args:
        description = r.curve_label
        with open(directory_path + "/{}.csv".format(description), "w") as datafile:

            writer = csv.writer(datafile, delimiter=",")

            for num in range(r.data_size):
                writer.writerow([r.x[num], r.y[num]])

        plt.step(r.x, r.y, **r.plotargs)
        plt.xlabel("Months")
        plt.ylabel("Proportion of Patients Alive")
    plt.legend()
    plt.savefig(directory_path + "/plot.pdf")
