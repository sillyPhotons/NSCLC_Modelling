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

    mu = np.log(Constants.TABLE2[stage][1])
    sigma = np.sqrt(2*(np.abs(np.log(Constants.TABLE2[stage][0]) - mu)))

    params = Parameters()
    params.add('mean_growth_rate', value=7*10**-5, min=0, vary=False)
    params.add('std_growth_rate', value=7.23*10**-3, min=0, vary=False)
    params.add('carrying_capacity',
               value=pop_manager.get_volume_from_diameter(30), min=0, vary=False)
    params.add('mean_tumor_diameter',
               value=mu,
               vary=False,
               min=Constants.REFER_TUMOR_SIZE_DIST[stage][2],
               max=Constants.REFER_TUMOR_SIZE_DIST[stage][3])
    params.add('std_tumor_diameter',
               value=sigma,
               vary=False,
               min=Constants.REFER_TUMOR_SIZE_DIST[stage][2],
               max=Constants.REFER_TUMOR_SIZE_DIST[stage][3])
    params.add("mean_rad_alpha", 
               value = Constants.RAD_ALPHA[0],
               vary = False,
               min = 0,
               max = np.inf)
    params.add("std_rad_alpha", 
               value=Constants.RAD_ALPHA[1],
               vary=False,
               min = 0,
               max = np.inf)

    dat = np.loadtxt("./Data/stage{}Better.csv".format(stage), delimiter=',')
    # dat = np.loadtxt("./Data/radiotherapy.csv", delimiter=',')
    x, data = rd.read_file(dat)
    x = np.around(x)

    # vdts = pp.predict_VDT(params, np.arange(
    #     sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION), pop_manager, m.discrete_time_tumor_volume_GENG)

    # plt.hist(vdts, 50, range = [0, 500],density=True, alpha=0.7, rwidth=0.85)
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
