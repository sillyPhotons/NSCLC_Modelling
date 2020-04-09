import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit

import Constants
import Model as m
import ReadData as rd
import ParallelModel as pm
import GetProperties as gp
from Result import ResultObj, ResultManager
from CostFunction import cost_function_no_treatment

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def run(cost_function, params, fcn_args):

    start = time.time()
    minner = Minimizer(cost_function, params, fcn_args)
    result = minner.minimize(method="powell")
    report_fit(result.params)
    end = time.time()
    runtime = end - start
    print("A Total of {} seconds to complete \U0001F606".format(runtime))

    return result


sampling_range = [0, 60]
monte_carlo_patient_size = 10000
pop_manager = gp.PropertyManager(monte_carlo_patient_size)
res_manager = ResultManager()

for stage in Constants.REFER_TUMOR_SIZE_DIST.keys():
# for stage in ["4"]:

    params = Parameters()
    params.add('mean_growth_rate', value=7*10**-5, min=0, vary=False)
    params.add('std_growth_rate', value=7.23*10**-3, min=0, vary=False)
    # params.add('carrying_capacity', value=30, min=0, vary=False)
    params.add('carrying_capacity',
               value=pop_manager.get_volume_from_diameter(30), min=0, vary=False)
    params.add('mean_tumor_diameter',
               value=Constants.REFER_TUMOR_SIZE_DIST[stage][0],
               vary=False,
               min=Constants.REFER_TUMOR_SIZE_DIST[stage][2],
               max=Constants.REFER_TUMOR_SIZE_DIST[stage][3])
    params.add('std_tumor_diameter',
               value=Constants.REFER_TUMOR_SIZE_DIST[stage][1],
               vary=False,
               min=Constants.REFER_TUMOR_SIZE_DIST[stage][2],
               max=Constants.REFER_TUMOR_SIZE_DIST[stage][3])

    dat = np.loadtxt("./Data/stage{}Better.csv".format(stage), delimiter=',')
    # dat = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
    x, data = rd.read_file(dat)
    x = np.around(x)

    # px, py = m.predict_KMSC_discrete(params, np.arange(
    #     sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION), pop_manager, m.rk4_tumor_volume)

    px, py = pm.predict_KMSC_discrete(params, np.arange(
        sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION), pop_manager, m.discrete_time_tumor_volume_GENG)

    res_manager.record_prediction(
        ResultObj(plt.step, x, data, "Months",
                  "Proportion of Patients Alive", curve_label="Stage {} Data".format(stage), label="Stage {} Data".format(stage), color="black", alpha=0.7),
        ResultObj(plt.step, px, py, "Months",
                  "Proportion of Patients Alive",
                  curve_label="Stage {} Model".format(stage),
                  label="Stage {} Model".format(stage), alpha=0.7),
        comment="stage_[{}]_Reproduction".format(stage)
    )
