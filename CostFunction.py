import numpy as np
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from Model import gompertz_ode, gompertz_analytical
import matplotlib.pyplot as plt
from Constants import DEATH_DIAMETER
import time


def cost_function_no_treatment(params, x, data, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    start = time.time()

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        cell_number = pop_manager.get_tumor_cell_number_from_diameter(
            tumor_diameter)

        solved_cell_number = gompertz_analytical(cell_number, x, growth_rate, carrying_capacity)
        
        # odeint(gompertz_ode, cell_number, x, args=(
        #     growth_rate, carrying_capacity))

        solved_diameter = pop_manager.get_diameter_from_tumor_cell_number(
            solved_cell_number)
        
        # plt.plot(x, solved_diameter)
        # plt.show()

        try:
            death_time = next(x for x, val in enumerate(solved_diameter)
                                if val >= DEATH_DIAMETER)

        except:
            death_time = None

        if (death_time is not None):
            patients_alive = [(patients_alive[num] - 1) if num >=
                                death_time else patients_alive[num] for num in range(len(x))]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    end = time.time()
    runtime = end - start
    print("Iteration completed in {} seconds.".format(runtime))

    # plt.plot(x,patients_alive)
    # plt.show()

    return (patients_alive - data)
