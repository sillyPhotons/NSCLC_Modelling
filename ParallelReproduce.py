"""
    ParallelReproduce.py: This is used to reproduce the results in Geng's paper

"""
import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit

import Constants
import Model as m
import ReadData as rd
import ParallelPredict as pp
import GetProperties as gp
from Result import ResultObj, ResultManager

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
ray.init()

sampling_range = [0, 60]
monte_carlo_patient_size = 10000
pop_manager = gp.PropertyManager(monte_carlo_patient_size)
res_manager = ResultManager()

for stage in Constants.REFER_TUMOR_SIZE_DIST.keys():
    # for stage in ["4"]:

    params = pop_manager.get_param_object_for_no_treatment(stage=stage)

    dat = np.loadtxt("./Data/stage{}Better.csv".format(stage), delimiter=',')
    # dat = np.loadtxt("./Data/radiotherapy.csv", delimiter=',')
    x, data = rd.read_file(dat)
    x = np.around(x)

    # vdts = pp.predict_VDT(params, np.arange(
    #     sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION), pop_manager, m.discrete_time_tumor_volume_GENG)

    # plt.hist(vdts, 50, range = [0, 500],density=True, alpha=0.7, rwidth=0.85, align='left')
    # plt.show()

    px, py = pp.predict_KMSC_discrete(params,
                                      np.arange(
                                          sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION),
                                      pop_manager,
                                      m.discrete_time_tumor_volume_GENG)

    res_manager.record_prediction(
        ResultObj(plt.step, x, data, "Months",
                  "Proportion of Patients Alive", curve_label="Stage {} Data".format(stage), label="Stage {} Data".format(stage), color="black", alpha=0.7),
        ResultObj(plt.step, px, py, "Months",
                  "Proportion of Patients Alive",
                  curve_label="Stage {} Model".format(stage),
                  label="Stage {} Model".format(stage), alpha=0.7),
        comment="Stage_[{}]".format(stage)
    )
