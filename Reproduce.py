import logging
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import time
from CostFunction import cost_function_no_treatment
from Model import gompertz_ode, predict_no_treatment, predict_volume_doubling_time, predict_no_treatment_diameter, predict_discrete_time_volume
import GetProperties as gp
import ReadData as rd
import matplotlib.pyplot as plt
from Result import ResultObj, ResultManager
import Constants

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

    # dat = np.loadtxt("./Data/stage{}.csv".format(stage), delimiter=',')
    dat = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
    x, data = rd.read_file(dat)

    px, py = predict_discrete_time_volume(params, np.arange(
        sampling_range[0], sampling_range[1]*31 + Constants.RESOLUTION, Constants.RESOLUTION), pop_manager)

    res_manager.record_prediction(
        ResultObj(plt.step, x, data, "Months",
        "Proportion of Patients Alive", curve_label="Stage {} Data".format(stage), label="Stage {} Data".format(stage), color="black", alpha=0.7),
        ResultObj(plt.step, px, py, "Months", 
        "Proportion of Patients Alive", 
        curve_label="Stage {} Model".format(stage), 
        label="Stage {} Model".format(stage), alpha=0.7),             
        comment="stage_[{}]_Reproduction".format(stage)
        )
