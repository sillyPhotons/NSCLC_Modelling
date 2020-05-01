"""
    Main.py, this is where all the defined classes and functions in the other
    files come together. 
"""
import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit

import Constants as c
import Model as m
import ReadData as rd
import CostFunction as cf
import GetProperties as gp
import ParallelPredict as pp
from Result import ResultObj, ResultManager

# configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M', level=logging.INFO)
ray.init()


def run(cost_function, params, fcn_args):
    """
    returns a `MinizerResult` object that represents the results of an 
    optimization algorithm

    `cost_function`: a callable (cost function) taking a `Parameters` object, 
        x numpy array, y numpy array, and other parameters
    `params`: `Parameters` object to be passed into the `cost_function`
    `fcn_args`: arugments for the `cost_function`
    """

    start = time.time()
    minner = Minimizer(cost_function, params, fcn_args)
    result = minner.minimize(method="powell")
    report_fit(result.params)
    end = time.time()
    runtime = end - start
    logging.critical(
        "A Total of {} seconds to complete \U0001F606".format(runtime))

    return result


# Determine the domain of KMSc curve. [0,60] means from month 0 to month 60
# this range should be converted to days when creating an array of this range
sampling_range = [0, 59]

# The number of patients to generate for the minization of the cost function
monte_carlo_patient_size = 1000

# Get an instance of a PropertyManager object, and initialize it patient size
pop_manager = gp.PropertyManager(monte_carlo_patient_size)

res_manager = ResultManager()

for stage in ["1"]:
    """
    Parameters object, we add Parameter objects to it, and we can specify
    whether that Parameter object can vary, and provide bounds to the value 
    of the estimated parameters

    We can specify:
        its `value`
        upper and lower of `value`: `min`, and `max`
        whether this varies during minimization `vary`
    """
    params = pop_manager.get_param_object_for_radiation()
    # Added so that it shows up in the report file
    params.add("Resolution", value=c.RESOLUTION, vary=False)

    params['corr'].vary = True
    # params['alpha_sigma'].vary = True
    # params.add('rho_mu', value=7.00*10**-5, min=0, vary=False)
    # params.add('rho_sigma', value=7.23*10**-3, min=0, vary=False)
    # params.add('K',
    #            value=pop_manager.get_volume_from_diameter(30),
    #            min=0,
    #            vary=False)
    # params.add('alpha_mu',
    #            value=c.RAD_ALPHA[0],
    #            vary=True,
    #            min=c.RAD_ALPHA[2],
    #            max=c.RAD_ALPHA[3])
    # params.add('alpha_sigma',
    #            value=c.RAD_ALPHA[1],
    #            vary=True,
    #            min=0 )
    # params.add("corr",
    #            value=c.GR_RS_CORRELATION,
    #            vary=False,
    #            min=-1,
    #            max=1)

    # x, data = rd.read_file("./Data/stage{}Better.csv".format(stage), interval=sampling_range)
    x, data = rd.read_file("./Data/radiotherapy.csv", interval=sampling_range)
    # around the x_coordinates of the data, convert to days
    x = np.around(x) * 31

    # Run minimization
    result = run(cf.cost_function_radiotherapy, params,
                 fcn_args=(x, data, pop_manager,
                           m.tumor_volume_GENG))

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
                                  ResultObj(plt.scatter, x/31., data, "Days",
                                            "Proportion of Patients Alive", curve_label="Stage {} Data".format(stage), label="Stage {} Data".format(stage), s = 25, alpha=0.7),
                                  ResultObj(plt.step, x/31., data + result.residual, "Days",
                                            "Proportion of Patients Alive", curve_label="{} Patient Model".format(monte_carlo_patient_size), label="{} Patient Model".format(monte_carlo_patient_size), alpha=0.7),
                                  # ResultObj(plt.step, px, py, "Months", "Proportion of Patients Alive",
                                  # curve_label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), label="{} Patients Model Prediction".format
                                  # (pop_manager2.get_patient_size()), alpha=0.7),
                                  comment="rad_alpha_corr_0_"
                                  )

# result.params["V_mu"].vary = True
# result.params["V_sigma"].vary = True
# result.params["rho_mu"].vary = False
# result.params["rho_sigma"].vary = False
# result.params["K"].vary = False

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
