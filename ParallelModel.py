import logging
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from ReadData import read_file
from Constants import DEATH_DIAMETER, RESOLUTION
import time
import ray

ray.init()

@ray.remote
def sim_patient(initial_volume, growth_rate, carrying_capacity, death_volume, num_steps, func_pointer):

    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume

    death_time = None
    for i in range(1, num_steps):

        cancer_volume[i] = func_pointer(
            cancer_volume[i - 1], growth_rate, carrying_capacity, h=RESOLUTION)

        if cancer_volume[i] > death_volume:
            cancer_volume[i] = death_volume
            death_time = i
            break
    
    if (death_time is not None):

        return death_time
    else:
        return None


def predict_KMSC_discrete(params, x, pop_manager, func_pointer):
    """
    """
    start = time.time()

    from scipy.stats import truncnorm

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    ######################################################################
    lowerbound = params['mean_tumor_diameter'].min
    upperbound = params['mean_tumor_diameter'].max

    initial_diameter = pop_manager.sample_lognormal_param(mean_tumor_diameter,
                                                          std_tumor_diameter,
                                                          retval=patient_size,
                                                          lowerbound=lowerbound,
                                                          upperbound=upperbound)
    ######################################################################

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
        mean=mean_growth_rate, std=std_growth_rate, retval=patient_size, lowerbound=0, upperbound=None)

    num_steps = x.size
    death_volume = pop_manager.get_volume_from_diameter(DEATH_DIAMETER)
   
    id_list = list()
    for num in range(patient_size):
        
        obj_id = sim_patient.remote(initial_volume[num], growth_rates[num], carrying_capacity, death_volume, num_steps, func_pointer)

        id_list.append(obj_id)
    
    death_times = [ray.get(obj_id) for obj_id in id_list]
    
    for times in death_times:
        if times is not None:
            patients_alive = [(patients_alive[k] - 1) if k >=
                                times else patients_alive[k] for k in range(num_steps)]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    end = time.time()
    runtime = end - start

    logging.info(
        "Minimization Iteration completed in {} seconds.".format(runtime))

    months = [num / 31. for num in x]

    return months, patients_alive