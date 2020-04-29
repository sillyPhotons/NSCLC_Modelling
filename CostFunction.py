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


def cost_function_radiotherapy(params, x, data, pop_manager, func_pointer):
    """
    Please verify that the desired radiotherapy fractionation is reflected in 
    variables of the `Constant.py` file.

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
    alpha_mu = p['alpha_mu']
    alpha_sigma = p['alpha_sigma']
    corr = p['corr']

    patient_size = pop_manager.get_patient_size()
    num_steps = int((x[-1] - x[0])/c.RESOLUTION)

    xsize = x.size
    patients_alive = [patient_size] * xsize

    initial_diameter = pop_manager.get_initial_diameters(
        stage_1=c.RADIATION_ONLY_PATIENT_PERCENTAGE["1"],
        stage_2=c.RADIATION_ONLY_PATIENT_PERCENTAGE["2"],
        stage_3A=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3A"],
        stage_3B=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3B"],
        stage_4=c.RADIATION_ONLY_PATIENT_PERCENTAGE["4"])

    initial_volume = pop_manager.get_volume_from_diameter(np.array(initial_diameter))

    alpha = np.array([alpha_mu, alpha_sigma, c.RAD_ALPHA[2], c.RAD_ALPHA[3]])
    rho = np.array(
        [rho_mu, rho_sigma, params['rho_mu'].min, params['rho_mu'].max])

    alpha_and_rho =\
        pop_manager.sample_correlated_params(alpha,
                                             rho,
                                             corr,
                                             retval=patient_size)

    treatment_delay = pop_manager.get_treatment_delay()
    treatment_days = pop_manager.get_radiation_days(treatment_delay, num_steps)
    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):
        obj_id =\
            pp.sim_death_time_with_radiation.remote(num_steps,
                                                 initial_volume[num],
                                                 death_volume,
                                                 treatment_days[num],
                                                 func_pointer,
                                                 alpha_and_rho[num, 1],
                                                 K,
                                                 alpha=alpha_and_rho[num, 0],
                                                 beta=alpha_and_rho[num, 0]/c.ALPHA_PER_BETA
                                                 )

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

