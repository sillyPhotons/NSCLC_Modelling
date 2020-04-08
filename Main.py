"""
    Main.py, this is where all the defined classes and functions in the other
    files come together. 
"""
import logging
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import time
from CostFunction import cost_function_no_treatment, cost_function_no_treatment_diameter, cost_function_no_treatment_volume, cost_function_discrete_time_volume
from Model import gompertz_ode, predict_no_treatment, predict_volume_doubling_time, predict_no_treatment_diameter
import GetProperties as gp
import ReadData as rd
import matplotlib.pyplot as plt
from Result import ResultObj, ResultManager

"""
    returns a MinizerResult object that represents the results of an 
    optimization algorithm

    cost_function: a callable (cost function) taking a Parameters object, 
        x numpy array, y numpy array, and other parameters
    params: Parameters object to be passed into the `cost_function`
    fcn_args: arugments for the cost_function
"""

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def run(cost_function, params, fcn_args):

    start = time.time()
    minner = Minimizer(cost_function, params, fcn_args)
    result = minner.minimize(method="powell")
    report_fit(result.params)
    end = time.time()
    runtime = end - start
    logging.critical("A Total of {} seconds to complete \U0001F606".format(runtime))

    return result


# Determine the domain of KMSc curve. [0,60] means from month 0 to month 60
sampling_range = [0, 60]
# The number of patients to generate for the minization of the cost function
monte_carlo_patient_size = 1000
pop_manager = gp.PropertyManager(monte_carlo_patient_size)
res_manager = ResultManager()

# Parameters object, we add Parameter objects to it, and we can specify whether
# that Parameter object can vary, and provide bounds to the value of the
# estimated parameters
params = Parameters()
params.add('mean_growth_rate', value=7.00*10**-5, min=0, vary=True)
params.add('std_growth_rate', value=7.23*10**-3, min=0, vary=True)
# params.add('carrying_capacity', value=30, min=0, vary=False)
params.add('carrying_capacity',
           value=pop_manager.get_volume_from_diameter(30), min=0, vary = False)
params.add('mean_tumor_diameter', value=1.72, vary=False, min=0, max=5)
params.add('std_tumor_diameter', value=4.70, vary=False, min=0, max=5)

# Read the data
dat = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
x, data = rd.read_file(dat)

result = run(cost_function_discrete_time_volume, params,
             fcn_args=(x, data, pop_manager))

# pop_manager2 = gp.PropertyManager(1432)
# px, py = predict_no_treatment_volume(
#     result.params, np.arange(sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager2)

# vdt = predict_volume_doubling_time(result.params, np.arange(
#     sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager2)

# plt.hist(vdt,100,density=True)
# plt.show()

# Passign ResultObj into the ResultManager object, where they are plotted and
# saved
res_manager.record_simulation(result,
                              ResultObj(plt.step, x, data, "Months",
                                        "Proportion of Patients Alive", curve_label="Data", label="Data", color="black", alpha=0.7),
                              ResultObj(plt.step, x, data + result.residual, "Months",
                                        "Proportion of Patients Alive", curve_label="{} Patient Model".format(monte_carlo_patient_size), label="{} Patient Model".format(monte_carlo_patient_size), alpha=0.7),
                              #ResultObj(plt.step, px, py, "Months", "Proportion of Patients Alive",
                                        #curve_label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), label="{} Patients Model Prediction".format
                                        #(pop_manager2.get_patient_size()), alpha=0.7),
                              comment="stage_[1]_DISCRETE"
                              )

# result.params["mean_tumor_diameter"].vary = True
# result.params["std_tumor_diameter"].vary = True
# result.params["mean_growth_rate"].vary = False
# result.params["std_growth_rate"].vary = False
# result.params["carrying_capacity"].vary = False

# pop_man = gp.PropertyManager(monte_carlo_patient_size)
# result2 = run(cost_function_no_treatment, result.params,
#               fcn_args=(x, data, pop_man))

# # px2, py2 = predict_no_treatment(
# #     result2.params, np.arange(sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager2)

# res_manager.record_simulation(result2,
#                               ResultObj(x, data, "Months",
#                                         "Proportion of Patients Alive", curve_label="Data", label="Data", color="black", alpha=0.7),
#                               ResultObj(x, data + result2.residual, "Months",
#                                         "Proportion of Patients Alive", curve_label="Model", label="Model", alpha=0.7),
#                               #ResultObj(px2, py2, "Months", "Proportion of Patients Alive",
#                                         #curve_label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), alpha=0.7),
#                               comment="stage_[1]_step_2"
#                               )
