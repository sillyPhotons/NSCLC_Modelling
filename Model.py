import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from ReadData import read_file
import GetProperties as gp

def gompertz_ode(N, t, growth_rate, carrying_capacity):

    dNdt = growth_rate*N*np.log(carrying_capacity/N)

    return dNdt


def predict_no_treatment(params, x, patient_num, stage):

    growth_rate = params['growth_rate']
    carrying_capacity = params['carrying_capacity']

    time = x
    patients_alive = [patient_num] * len(time)

    for num in range(patient_num):
        
        cell_number = gp.get_tumor_cell_number_from_diameter(gp.sample_dist(stage, 1))

        solved_cell_number = odeint(gompertz_ode, cell_number, time, args=(
            growth_rate, carrying_capacity))
        
        solved_diameter = gp.get_diameter_from_tumor_cell_number(solved_cell_number)
        
        try:
            death_time = next(x for x, val in enumerate(solved_diameter) 
                                  if val >= 13) 
            # print(death_time)
        except:
          death_time = None
    
        if (death_time is not None):
            patients_alive = [(patients_alive[num] - 1) if num >= death_time else patients_alive[num] for num in range(len(time))]

    patients_alive = np.array(patients_alive)

    
    patients_alive = patients_alive/patients_alive[0]
    
    return time, patients_alive