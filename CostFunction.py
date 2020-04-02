import numpy as np
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from Model import gompertz_ode
from ReadData import read_file
import matplotlib.pyplot as plt
import GetProperties as gp
from Constants import DEATH_DIAMETER


def cost_function_no_treatment(params, x, data,
                               file_path="./population1.csv"):

    growth_rate = params['growth_rate']
    carrying_capacity = params['carrying_capacity']

    dat = np.loadtxt(file_path, delimiter=',')
    patient_num, tumor_diameter = read_file(dat)
    
    cell_number = gp.get_tumor_cell_number_from_diameter(tumor_diameter)

    time = x
  
    patients_alive = [len(patient_num)] * len(time)

    for num in range(len(patient_num)):
        
        solved_cell_number = odeint(gompertz_ode, cell_number[num], time, args=(
            growth_rate, carrying_capacity))
        
        solved_diameter = gp.get_diameter_from_tumor_cell_number(solved_cell_number)
        
        try:
            death_time = next(x for x, val in enumerate(solved_diameter) 
                                  if val >= DEATH_DIAMETER) 
            # print(death_time)
        except:
          death_time = None
    
        if (death_time is not None):
            patients_alive = [(patients_alive[num] - 1) if num >= death_time else patients_alive[num] for num in range(len(time))]

    patients_alive = np.array(patients_alive)

    
    patients_alive = patients_alive/patients_alive[0]
    
    # plt.step(time, patients_alive)
    # plt.show()

    print("Iteration Complete.")
    return (patients_alive - data)
