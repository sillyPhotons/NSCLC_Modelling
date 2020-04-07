import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import time
from CostFunction import cost_function_no_treatment
from Model import gompertz_ode, predict_no_treatment, predict_volume_doubling_time, predict_no_treatment_diameter
import GetProperties as gp
import ReadData as rd
import matplotlib.pyplot as plt
from Result import ResultObj, ResultManager
import Constants

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
monte_carlo_patient_size = 1000
pop_manager = gp.PropertyManager(monte_carlo_patient_size)
res_manager = ResultManager()

params = Parameters()
params.add('mean_growth_rate', value=7.00*10**-5*np.pi**3, min=0, vary=False)
params.add('std_growth_rate', value=7.23*10**-3*np.pi**3, min=0, vary=False)
params.add('carrying_capacity',
           value=pop_manager.get_tumor_cell_number_from_diameter(30), min=0, vary=False)
params.add('mean_tumor_diameter', value=1.72, vary=False, min=0, max=5)
params.add('std_tumor_diameter', value=4.70, vary=False, min=0, max=5)

dat = np.loadtxt("./Data/stage1Better.csv", delimiter=',')
x, data = rd.read_file(dat)

px, py = predict_no_treatment(
    params, np.arange(sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager)

# vdt = predict_volume_doubling_time(params, np.arange(
#     sampling_range[0], sampling_range[1] + 0.1, 0.1), pop_manager)

# plt.hist(vdt,100,density=True)
# plt.show()

res_manager.record_prediction(ResultObj(plt.step, x, data, "Months",
                                        "Proportion of Patients Alive", curve_label="Data", label="Data", color="black", alpha=0.7),
                              ResultObj(plt.step, px, py, "Months", "Proportion of Patients Alive",
                                        curve_label="{} Patient Paper Reproduction".format(pop_manager.get_patient_size()), label="{} Patient Paper Reproduction".format(pop_manager.get_patient_size()), alpha=0.7),
                              comment="stage_[1]_Reproduction"
                              )
