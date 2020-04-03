import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from ReadData import read_file
from Constants import DEATH_DIAMETER

def gompertz_ode(N, t, growth_rate, carrying_capacity):

    dNdt = growth_rate*N*np.log(carrying_capacity/N)

    return dNdt


def predict_no_treatment(params, x, pop_manager, stage):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)

        cell_number = pop_manager.get_tumor_cell_number_from_diameter(
            tumor_diameter[0])

        solved_cell_number = odeint(gompertz_ode, cell_number, x, args=(
            growth_rate, carrying_capacity))

        solved_diameter = pop_manager.get_diameter_from_tumor_cell_number(
            solved_cell_number)

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

    return x, patients_alive
