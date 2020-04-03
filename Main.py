import numpy as np
from lmfit import Minimizer, Parameters, report_fit, fit_report
import time
from CostFunction import cost_function_no_treatment
from Model import gompertz_ode, predict_no_treatment
import GetProperties as gp
import ReadData as rd


def record_simulation (result, *args):
    
    from datetime import datetime
    import os
    
    now = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    directory_path = "./Results/sim_{}".format(now)
    os.mkdir(directory_path)
    
    with open(directory_path + "/report.txt", "w") as report:
        report.write(fit_report(result))
        
monte_carlo_patient_size = 1
pop_manager = gp.PropertyManager(monte_carlo_patient_size)

params = Parameters()
params.add('mean_growth_rate', value=7.00*10**-5, min=0)
params.add('std_growth_rate', value=7.23*10**-3, min=0)
params.add('carrying_capacity',
           value=pop_manager.get_tumor_cell_number_from_diameter(30), min=0)
params.add('mean_tumor_diameter', value=2.5, vary=False)
params.add('std_tumor_diameter', value=2.5, vary=False)

sampling_interval = 0.5  # a data point every 0.5 months
x, data = rd.get_data("./Data/stage1.csv", sampling_interval, range=[0, 120])

start = time.time()
minner = Minimizer(cost_function_no_treatment, params,
                   fcn_args=(x, data, pop_manager))

result = minner.minimize(method="powell")
end = time.time()
runtime = end - start

print("Took {} seconds to complete".format(runtime))

final = data + result.residual

record_simulation(result)

# report_fit(result)

# px500, py500 = predict_no_treatment(
#     result.params, np.arange(0, 60 + 0.1, 0.1), pop_manager, 1)
# px1000, py1000 = predict_no_treatment(
#     result.params, np.arange(0, 60 + 0.1, 0.1), pop_manager, 1)
# # px10000, py10000 = predict_no_treatment(
# #     result.params, np.arange(0, 60 + 0.1, 0.1), pop_manager, 1)

# try:
#     import matplotlib.pyplot as plt
#     import matplotlib as mpl
#     mpl.rcParams["font.family"] = "FreeSerif"
#     plt.rc("text", usetex=True)
#     plt.figure(dpi=100)

#     plt.step(x, data, 'k+', label="Data")
#     plt.step(x, final, label="Data + Residual")
#     # plt.step(px500, py500, label="500 Patient Prediction Model")
#     # plt.step(px1000, py1000, label="1000 Patient Prediction Model")
#     # plt.step(px10000, py10000, label = "10000 Patient Prediction Model")

#     plt.xlabel("Months")
#     plt.ylabel("Proportion of Patients Alive")
#     plt.legend()
#     plt.savefig("Stage_[1]_[{}]Patients_[{}]Sampling_Interval.pdf".format(
#         monte_carlo_patient_size, sampling_interval))
# except ImportError:
#     pass
