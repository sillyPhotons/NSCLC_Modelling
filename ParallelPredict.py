"""
Implementations of functions that predict
    - Volume Doubling time
    - KMSC curve
that involves the use of some sort of concurrency
"""

import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters

from ReadData import read_file
from Constants import DEATH_DIAMETER, RESOLUTION, SURVIVAL_REDUCTION
import Model as m



@ray.remote
def sim_patient_death_time(initial_volume, growth_rate, carrying_capacity, death_volume, num_steps, func_pointer):
    """
    This function is decorated with `@ray.remote`, which means that it is a funciton that may be called multiple times in parallel. Given the parameters, returns a single integer equal to number of time steps taken before the patient tumor volume exceeds `death_volume`

    `initial_volume`: the initial tumor volume
    `growth_rate`: floating point value
    `carrying_capacity`: floating point value
    `death_volume`: volume of tumor at which point the patient is considered dead
    `num_steps`: number of `RESOLUTION` steps to take
    `func_pointer`: discrete time model of the mode taking `initial_volume`, `growth_rate`, `carrying_capacity` 
    """

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
    Returns the x,y series to plot the KMSC for a patient population. x has units of months, and y is the proportion of patients alive. Every y value is reduced by `SURVIVAL_REDUCTION` found in `Constants.py`, except the point at x = 0.

    `param`: `Parameters` object 
    `x`: time array to find the KMSC curve on
    `pop_manager`: `PropertyManager` object
    `func_pointer`: a function object. In python, functions are first class objects. Discrete time model of the mode taking `initial_volume`, `growth_rate`, `carrying_capacity` 

    Requires:
        `Parameters` object contains Parameter objects with the keys:
            `mean_growth_rate`
            `std_growth_rate`
            `carrying_capacity`
            `mean_tumor_diameter`
            `std_tumor_diameter`
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
    num_steps = x.size
    patients_alive = [patient_size] * num_steps

    ######################################################################
    lowerbound = (np.log(params['mean_tumor_diameter'].min) -
                  mean_tumor_diameter) / std_tumor_diameter
    upperbound = (np.log(params['mean_tumor_diameter'].max) -
                  mean_tumor_diameter) / std_tumor_diameter

    norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=patient_size)

    initial_diameter = list(np.exp(
        (norm_rvs * std_tumor_diameter) + mean_tumor_diameter))
    ######################################################################

    # ######################################################################
    # lowerbound = params['mean_tumor_diameter'].min
    # upperbound = params['mean_tumor_diameter'].max

    # lognormal_sigma = np.sqrt(np.log((std_tumor_diameter**2)/(mean_tumor_diameter**2) + 1))

    # lognormal_mean = np.log(mean_tumor_diameter) - (lognormal_sigma**2)/2.

    # initial_diameter = pop_manager.sample_lognormal_param(lognormal_mean,
    #                                                       lognormal_sigma,
    #                                                       retval=patient_size,
    #                                                       lowerbound=lowerbound,
    #                                                       upperbound=upperbound)
    # ######################################################################

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
        mean=mean_growth_rate, std=std_growth_rate, retval=patient_size, lowerbound=0, upperbound=None)
    
    death_volume = pop_manager.get_volume_from_diameter(DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_death_time.remote(
            initial_volume[num], growth_rates[num], carrying_capacity, death_volume, num_steps, func_pointer)

        id_list.append(obj_id)

    logging.info("Patient simulation complete, creating survival curve.")
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
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    months = [num / 31. if num ==
              0 else (num / 31.)*(1 - SURVIVAL_REDUCTION/100.) for num in x]

    return months, patients_alive

@ray.remote
def sim_patient_one_year(initial_volume, growth_rate, carrying_capacity, death_volume, num_steps, func_pointer):
    """
    This function is decorated with `@ray.remote`, which means that it is a funciton that may be called multiple times in parallel. Given the parameters, returns a single integer equal to the volume doubling time of the patient

    `initial_volume`: the initial tumor volume
    `growth_rate`: floating point value
    `carrying_capacity`: floating point value
    `death_volume`: volume of tumor at which point the patient is considered dead
    `num_steps`: number of `RESOLUTION` steps to take until 365 days
    `func_pointer`: function object, discrete time model of the mode taking `initial_volume`, `growth_rate`, `carrying_capacity` 
    """
    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume

    for i in range(1, num_steps):

        cancer_volume[i] = func_pointer(
            cancer_volume[i - 1], growth_rate, carrying_capacity, h=RESOLUTION)

    return m.volume_doubling_time(initial_volume, cancer_volume[-1])

def predict_VDT(params, x, pop_manager, func_pointer):
    """
    Returns a numpy array of volume doubling time entries for a patient population in days.

    `param`: `Parameters` object 
    `x`: time array to find the KMSC curve on
    `pop_manager`: `PropertyManager` object
    `func_pointer`: a function object. In python, functions are first class objects. Discrete time model of the mode taking `initial_volume`, `growth_rate`, `carrying_capacity` 
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

    ######################################################################
    lowerbound = (np.log(params['mean_tumor_diameter'].min) -
                  mean_tumor_diameter) / std_tumor_diameter
    upperbound = (np.log(params['mean_tumor_diameter'].max) -
                  mean_tumor_diameter) / std_tumor_diameter

    norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=patient_size)

    initial_diameter = list(np.exp(
        (norm_rvs * std_tumor_diameter) + mean_tumor_diameter))
    ######################################################################

    # ######################################################################
    # lowerbound = params['mean_tumor_diameter'].min
    # upperbound = params['mean_tumor_diameter'].max

    # initial_diameter = pop_manager.sample_lognormal_param(mean_tumor_diameter,
    #                                                       std_tumor_diameter,
    #                                                       retval=patient_size,
    #                                                       lowerbound=lowerbound,
    #                                                       upperbound=upperbound)
    # ######################################################################
    
    # ######################################################################
    # lowerbound = params['mean_tumor_diameter'].min
    # upperbound = params['mean_tumor_diameter'].max

    # lognormal_sigma = np.sqrt(np.log((std_tumor_diameter**2)/(mean_tumor_diameter**2) + 1))

    # lognormal_mean = np.log(mean_tumor_diameter) - (lognormal_sigma**2)/2.

    # initial_diameter = pop_manager.sample_lognormal_param(lognormal_mean,
    #                                                       lognormal_sigma,
    #                                                       retval=patient_size,
    #                                                       lowerbound=lowerbound,
    #                                                       upperbound=upperbound)
    # ######################################################################

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
        mean=mean_growth_rate, std=std_growth_rate, retval=patient_size, lowerbound=0, upperbound=None)


    death_volume = pop_manager.get_volume_from_diameter(DEATH_DIAMETER)
    steps_to_one_year = int(365./RESOLUTION)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_one_year.remote(initial_volume[num], growth_rates[num], carrying_capacity, death_volume, steps_to_one_year, func_pointer)
        
        id_list.append(obj_id)

    logging.info("Patient simulation complete, fetching volume doubling times.")
    vdts = [ray.get(obj_id) for obj_id in id_list]

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 VDT prediction completed in {} seconds.".format(runtime))

    return np.array(vdts)