import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import time
from CostFunction import cost_function_no_treatment
from Model import gompertz_ode, predict_no_treatment, predict_volume_doubling_time
import GetProperties as gp
import ReadData as rd
import matplotlib.pyplot as plt
from Result import ResultObj, ResultManager


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
monte_carlo_patient_size = 2000
pop_manager = gp.PropertyManager(monte_carlo_patient_size)
res_manager = ResultManager()

params = Parameters()
params.add('mean_growth_rate', value=7.00*10**-5, min=0, vary=True)
params.add('std_growth_rate', value=7.23*10**-3, min=0, vary=True)
params.add('carrying_capacity',
           value=pop_manager.get_tumor_cell_number_from_diameter(30), min=0)
params.add('mean_tumor_diameter', value=2.5, vary=False, min=0, max=5)
params.add('std_tumor_diameter', value=2.5, vary=False, min=0, max=5)

sampling_interval = 0.5  # a data point every 0.5 months
x, data = rd.get_data("./Data/stage1.csv",
                      sampling_interval, range=sampling_range)

result = run(cost_function_no_treatment, params,
             fcn_args=(x, data, pop_manager))

pop_manager2 = gp.PropertyManager(1432)
# px, py = predict_no_treatment(
#     result.params, np.arange(sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager2)
vdt = predict_volume_doubling_time(result.params, np.arange(
    sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager2)

plt.hist(vdt,100,density=True)
plt.show()

# res_manager.record_simulation(result,
#                               ResultObj(plt.plot, x, data, "Months",
#                                         "Proportion of Patients Alive", curve_label="Data", label="Data", color="black", alpha=0.7),
#                               ResultObj(plt.plot, x, data + result.residual, "Months",
#                                         "Proportion of Patients Alive", curve_label="Model", label="Model", alpha=0.7),
#                               # ResultObj(px, py, "Months", "Proportion of Patients Alive",
#                               # curve_label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), label="{} Patients Model Prediction".format(pop_manager2.get_patient_size()), alpha=0.7),
#                               comment="stage_[1]"
#                               )

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
