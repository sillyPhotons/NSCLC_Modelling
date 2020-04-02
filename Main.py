import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import time

from Initialize import create_patient_population
from CostFunction import cost_function_no_treatment
from Model import gompertz_ode, predict_no_treatment
import GetProperties as gp
import ReadData as rd



monte_carlo_patient_size = 50

params = Parameters()
params.add('growth_rate', value=7.00*10**-5, min=0)
params.add('carrying_capacity',
           value=gp.get_tumor_cell_number_from_diameter(30), min=0)

create_patient_population(num_patients=monte_carlo_patient_size)

sampling_interval = 0.5  # a data point every 0.5 months
x, data = rd.get_data("./Data/stage1.csv", sampling_interval, range = [0, 60])

start = time.time()
minner = Minimizer(cost_function_no_treatment, params,
                   fcn_args=(x, data, "./population1.csv"))

result = minner.minimize(method="powell")
end = time.time()
runtime = end - start

print("Took {} seconds to complete".format(runtime))

final = data + result.residual
report_fit(result)

px500, py500 = predict_no_treatment(result.params, np.arange(0, 60 + 0.1, 0.1), 500, 1)
px1000, py1000 = predict_no_treatment(result.params, np.arange(0, 60 + 0.1, 0.1), 1000, 1)
px10000, py10000 = predict_no_treatment(result.params, np.arange(0, 60 + 0.1, 0.1), 10000, 1)

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "FreeSerif"
    plt.rc("text", usetex=True)
    plt.figure(dpi=100)

    plt.step(x, data, 'k+', label = "Data")
    plt.step(x, final, label = "Data + Residual")
    plt.step(px500, py500, label = "500 Patient Prediction Model")
    plt.step(px1000, py1000, label = "1000 Patient Prediction Model")
    plt.step(px10000, py10000, label = "10000 Patient Prediction Model")
    
    plt.xlabel("Months")
    plt.ylabel("Proportion of Patients Alive")
    plt.legend()
    plt.savefig("Stage_[1]_[{}]Patients_[{}]Sampling_Interval.pdf".format(monte_carlo_patient_size, sampling_interval))
except ImportError:
    pass
