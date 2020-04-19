"""
Author: Ruiheng Su 2020

File containing the cost function for parameter estimation. 
"""
import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import Model as m
import Constants as c
import ParallelPredict as pp

def cost_function(params, x, data, pop_manager, func_pointer):
    """
    Returns the residual of model and data to be used by the `minimize` method 
    of the `lmfit` module, given a `Parameters` object, the x and y series of a 
    KMSC as numpy arrays, a `PropertyManager` object, and a pointer to a discrete time stepping function.

    Params::
        `param`: `Parameters` object 
        
        `x`: values on the time axis associated with each element of `data`, a numpy array

        `data`: y values of KMSC curve data, a numpy array
        
        `pop_manager`: `PropertyManager` object

        `func_pointer`: a discrete time model function object. (In python, functions are first class objects.) 
    """
    start = time.time()

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    V_mu = p['V_mu']
    V_sigma = p['V_sigma']

    patient_size = pop_manager.get_patient_size()
    num_steps = int((x[-1] - x[0])/c.RESOLUTION)

    xsize = x.size
    patients_alive = [patient_size] * xsize

    initial_diameter = pop_manager.sample_lognormal_param(
        V_mu, V_sigma, retval=patient_size, lowerbound=params['V_mu'].min, upperbound=params['V_mu'].max)

    initial_volume = pop_manager.get_volume_from_diameter(initial_diameter)

    growth_rates = pop_manager.sample_normal_param(
        mean=rho_mu, std=rho_sigma, retval=patient_size, lowerbound=0, upperbound=None)

    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):
        obj_id = pp.sim_patient_death_time.remote( num_steps, 
            initial_volume[num], death_volume, func_pointer, growth_rates[num], K)

        id_list.append(obj_id)

    logging.info("Patient simulation complete, creating survival curve.")
    death_times = [ray.get(obj_id) for obj_id in id_list]

    for times in death_times:
        if times is not None:
            patients_alive = [(patients_alive[k] - 1) if x[k] >=
                              times * c.RESOLUTION else patients_alive[k] for k in range(xsize)]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    patients_alive = np.array(patients_alive)
    patients_alive = (
        patients_alive/patients_alive[0]) * (1 - c.SURVIVAL_REDUCTION/100.)
    patients_alive[0] = 1.

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)
