"""
    This file contains the cost function. 
"""
import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.integrate import odeint
from lmfit import minimize, Parameters

import Model as m
import ParallelPredict as pp
from Constants import DEATH_DIAMETER, RESOLUTION, SURVIVAL_REDUCTION


def cost_function(params, x, data, pop_manager, func_pointer):
    """
    Returns the residual of model and data, given a Parameters object, the xy data for a KMSc, a `PropertyManager` object, and a pointer to a discrete time stepping function.

    `param`: `Parameters` object 
    `x`: x values of `data`
    `data`: y values of KMSC curve data
    `pop_manager`: `PropertyManager` object
    `func_pointer`: a function object. In python, functions are first class objects. Discrete time model of the mode taking `initial_volume`, `growth_rate`, `carrying_capacity` 

    Requires: 
        `Parameters object` contains Parameter objects with the keys:
            `mean_growth_rate`
            `std_growth_rate`
            `carrying_capacity`
            `mean_tumor_diameter`
            `std_tumor_diameter`
        The KMSc data at t = 0 must equal 1
        x,y series are numpy arrays
    """

    start = time.time()

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    num_steps = int((x[-1] - x[0])/RESOLUTION)

    xsize = x.size
    patients_alive = [patient_size] * xsize

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
        obj_id = pp.sim_patient_death_time.remote(
            initial_volume[num], growth_rates[num], carrying_capacity, death_volume, num_steps, func_pointer)

        id_list.append(obj_id)

    logging.info("Patient simulation complete, creating survival curve.")
    death_times = [ray.get(obj_id) for obj_id in id_list]

    for times in death_times:
        if times is not None:
            patients_alive = [(patients_alive[k] - 1) if x[k] >=
                              times * RESOLUTION else patients_alive[k] for k in range(xsize)]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    patients_alive = np.array(patients_alive)
    patients_alive = (
        patients_alive/patients_alive[0]) * (1 - SURVIVAL_REDUCTION/100.)
    patients_alive[0] = 1.

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)
