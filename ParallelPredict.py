"""
Author: Ruiheng Su 2020

Implementation of Geng's no-treatment model, or radiation only model. 
"""

import ray
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from ReadData import read_file
import Constants as c
import Model as m


@ray.remote
def sim_patient_death_time(num_steps, initial_volume, death_volume, func_pointer, *func_args, **func_kwargs):
    """
    Returns a scalar value representing the number of time steps before the 
    patient dies if he or she is not given any treatment. If the patient 
    recovers, or do not die within the simulationed time steps, then `None` is 
    returned. 

    This function is decorated with `@ray.remote`, which means that it is a 
    funciton that may be called multiple times in parallel.

    Params::
        `num_steps`: number of time steps to take

        `initial_volume`: the initial tumor volume

        `death_volume`: volume of tumor at which point the patient is considered dead

        `func_pointer`: discrete time model of the model taking `*func_args` and `**func_kwargs` as parameters
    """

    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume

    recover_prob = np.random.rand(num_steps)

    death_time = None
    for i in range(1, num_steps):

        cancer_volume[i] = func_pointer(
            cancer_volume[i - 1], *func_args, **func_kwargs)

        if cancer_volume[i] > death_volume:
            cancer_volume[i] = death_volume
            death_time = i
            break

        # probability that the tumor was controlled
        if recover_prob[i] < np.exp(-cancer_volume[i] * c.TUMOR_DENSITY):
            return None

    if (death_time is not None):

        return death_time
    else:
        return None


@ray.remote
def sim_death_time_with_radiation(num_steps, initial_volume, death_volume, treatment_days, func_pointer, *func_args, **func_kwargs):
    """
    Returns a scalar value representing the number of time steps before the 
    patient dies if he or she recieves a single radiotherapy treatement. If the 
    patient recovers, or do not die within the simulationed time steps, then 
    `None` is returned.

    This function is decorated with `@ray.remote`, which means that it is a 
    funciton that may be called multiple times in parallel.

    Params::
        `num_steps`: number of time steps to take

        `initial_volume`: the initial tumor volume

        `death_volume`: volume of tumor at which point the patient is considered dead

        `func_pointer`: discrete time model of the model taking `*func_args` and `**func_kwargs` as parameters
    """

    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume
    recover_prob = np.random.rand(num_steps)

    death_time = None
    for i in range(1, num_steps):

        cancer_volume[i] = func_pointer(
            cancer_volume[i - 1], *func_args, **func_kwargs, dose_step=treatment_days[i-1])

        if cancer_volume[i] > death_volume:
            cancer_volume[i] = death_volume
            death_time = i
            break

        # probability that the tumor was controlled
        if recover_prob[i] < np.exp(-cancer_volume[i] * c.TUMOR_DENSITY):
            return None

    if (death_time is not None):

        return death_time
    else:
        return None


@ray.remote
def sim_patient_radiation_response(num_steps, initial_volume, treatment_days, func_pointer, *func_args, **func_kwargs):
    """
    Returns an array representing the patient tumor volume at each time step. 

    This function is decorated with `@ray.remote`, which means that it is a 
    funciton that may be called multiple times in parallel.

    Params::
        `num_steps`: number of time steps to take

        `initial_volume`: the initial tumor volume

        `func_pointer`: discrete time model of the model taking `*func_args` and `**func_kwargs` as parameters
    """

    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume
    recover_prob = np.random.rand(num_steps)
    
    recover = False
    for i in range(1, num_steps):

        if recover:
            cancer_volume[i] = np.nan
        else:
            cancer_volume[i] = func_pointer(
                cancer_volume[i - 1], *func_args, **func_kwargs, dose_step=treatment_days[i-1])
        
            if recover_prob[i] < np.exp(-cancer_volume[i] * c.TUMOR_DENSITY):
                recover = True
    return cancer_volume


def Radiation_Response(V0, rho, K, alpha, beta, delay_days, x, pop_manager, func_pointer):
    """
    """
    start = time.time()
    num_steps = int(x.size + x[0]/c.RESOLUTION)

    treatment_delay = np.array(delay_days)
    treatment_days = pop_manager.get_radiation_days(treatment_delay, num_steps)

    obj_id =\
        sim_patient_radiation_response.remote(num_steps,
                                              V0,
                                              treatment_days[0],
                                              func_pointer,
                                              rho,
                                              K,
                                              alpha=alpha,
                                              beta=beta
                                              )

    patient_array = ray.get(obj_id)
    patient_array = patient_array[int(
        x[0]/c.RESOLUTION):int(x[-1]/c.RESOLUTION)+1]

    end = time.time()
    runtime = end - start

    logging.info("Patient treatment response simulation completed in {} seconds.".format(runtime))
    
    return x/31., patient_array

def Radiation_Treatment_Response_Multiple(params, x, pop_manager, func_pointer):
    """
    """
    start = time.time()  # start timing

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    alpha_mu = p['alpha_mu']
    alpha_sigma = p['alpha_sigma']

    patient_size = pop_manager.get_patient_size()
    num_steps = int(x.size + x[0]/c.RESOLUTION)

    initial_diameters = pop_manager.get_initial_diameters(
        stage_1=c.RADIATION_ONLY_PATIENT_PERCENTAGE["1"],
        stage_2=c.RADIATION_ONLY_PATIENT_PERCENTAGE["2"],
        stage_3A=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3A"],
        stage_3B=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3B"],
        stage_4=c.RADIATION_ONLY_PATIENT_PERCENTAGE["4"])

    initial_volume =\
        pop_manager.get_volume_from_diameter(np.array(initial_diameters))

    alpha = np.array([alpha_mu, alpha_sigma, c.RAD_ALPHA[2], c.RAD_ALPHA[3]])
    rho = np.array(
        [rho_mu, rho_sigma, params['rho_mu'].min, params['rho_mu'].max])

    alpha_and_rho =\
        pop_manager.sample_correlated_params(alpha,
                                             rho,
                                             c.GR_RS_CORRELATION,
                                             retval=patient_size)

    treatment_days = pop_manager.get_radiation_days(num_steps)
    
    id_list = list()
    for num in range(patient_size):

        obj_id =\
            sim_patient_radiation_response.remote(num_steps,
                                                 initial_volume[num],
                                                 treatment_days[num],
                                                 func_pointer,
                                                 alpha_and_rho[num, 1],
                                                 K,
                                                 alpha=alpha_and_rho[num, 0],
                                                 beta=alpha_and_rho[num, 0]/10.
                                                 )

        id_list.append(obj_id)

    logging.info("Patient treatment response simulation complete.")
    
    tumor_volume = list()
    for obj_id in id_list:
        patient_array = ray.get(obj_id)
        patient_array = patient_array[int(
        x[0]/c.RESOLUTION):int(x[-1]/c.RESOLUTION)+1]
        tumor_volume.append(patient_array)

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return x/31., tumor_volume

def KMSC_No_Treatment(params, x, pop_manager, func_pointer):
    """
    Returns the x,y series to plot the KMSC for a patient population. x has units of months, and y is the proportion of patients alive. Every y value is reduced by `SURVIVAL_REDUCTION` found in `Constants.py`, except the point at x = 0.

    `param`: `Parameters` object 
    `x`: numpy time array to find the KMSC curve on
    `pop_manager`: `PropertyManager` object
    `func_pointer`: a function object. In python, functions are first class objects. Discrete time model of the mode taking `initial_volume`, `growth_rate`, `K` 

    Requires:
        `Parameters` object contains Parameter objects with the keys:
            `rho_mu`
            `rho_sigma`
            `K`
            `V_mu`
            `V_sigma`
    """

    start = time.time()

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    V_mu = p['V_mu']
    V_sigma = p['V_sigma']

    patient_size = pop_manager.get_patient_size()
    num_steps = int(x.size + x[0]/c.RESOLUTION)
    patients_alive = [patient_size] * num_steps

    initial_diameter = pop_manager.sample_lognormal_param(
        V_mu, V_sigma, retval=patient_size, lowerbound=params['V_mu'].min, upperbound=params['V_mu'].max)

    initial_volume = pop_manager.get_volume_from_diameter(initial_diameter)

    growth_rates = pop_manager.sample_normal_param(
        mean=rho_mu, std=rho_sigma, retval=patient_size, lowerbound=0, upperbound=None)

    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_death_time.remote(num_steps,
                                               initial_volume[num],
                                               death_volume, func_pointer, growth_rates[num], K)

        id_list.append(obj_id)

    logging.info("Patient simulation complete, creating survival curve.")
    death_times = [ray.get(obj_id) for obj_id in id_list]

    for times in death_times:
        if times is not None:
            patients_alive = [(patients_alive[k] - 1) if k >=
                              times else patients_alive[k] for k in range(num_steps)]

    patients_alive = np.array(patients_alive)
    patients_alive = (
        patients_alive/patients_alive[0])*(1 - c.SURVIVAL_REDUCTION/100.)
    patients_alive[0] = 1.

    patients_alive = patients_alive[int(
        x[0]/c.RESOLUTION):int(x[-1]/c.RESOLUTION)+1]
    months = x/31.

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return months, patients_alive


@ray.remote
def sim_patient_one_year(num_steps, initial_volume, death_volume, func_pointer, *func_args, **func_kwargs):
    """
    This function is decorated with `@ray.remote`, which means that it is a funciton that may be called multiple times in parallel. Given the parameters, returns a single integer equal to the volume doubling time of the patient

    `initial_volume`: the initial tumor volume
    `growth_rate`: floating point value
    `K`: floating point value
    `death_volume`: volume of tumor at which point the patient is considered dead
    `num_steps`: number of `RESOLUTION` steps to take until 365 days
    `func_pointer`: function object, discrete time model of the mode taking `initial_volume`, `growth_rate`, `K` 
    """
    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume

    for i in range(1, num_steps):

        cancer_volume[i] = func_pointer(
            cancer_volume[i - 1], *func_args, **func_kwargs)

    return m.volume_doubling_time(initial_volume, cancer_volume[-1])


def predict_VDT(params, x, pop_manager, func_pointer):
    """
    Returns a numpy array of volume doubling time entries for a patient population in days.

    `param`: `Parameters` object 
    `x`: time array to find the KMSC curve on
    `pop_manager`: `PropertyManager` object
    `func_pointer`: a function object. In python, functions are first class objects. Discrete time model of the mode taking `initial_volume`, `growth_rate`, `K` 
    """

    start = time.time()

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    V_mu = p['V_mu']
    V_sigma = p['V_sigma']

    patient_size = pop_manager.get_patient_size()

    initial_diameter = pop_manager.sample_lognormal_param(
        V_mu, V_sigma, retval=patient_size, lowerbound=params['V_mu'].min, upperbound=params['V_mu'].max)

    initial_volume = pop_manager.get_volume_from_diameter(initial_diameter)

    growth_rates = pop_manager.sample_normal_param(
        mean=rho_mu, std=rho_sigma, retval=patient_size, lowerbound=0, upperbound=None)

    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)
    steps_to_one_year = int(365./c.RESOLUTION)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_one_year.remote(steps_to_one_year,
                                             initial_volume[num], death_volume, func_pointer, growth_rates[num], K)

        id_list.append(obj_id)

    logging.info(
        "Patient simulation complete, fetching volume doubling times.")
    vdts = [ray.get(obj_id) for obj_id in id_list]

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 VDT prediction completed in {} seconds.".format(runtime))

    return np.array(vdts)


def KMSC_With_Radiotherapy(params, x, pop_manager, func_pointer):

    start = time.time()  # start timing

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    alpha_mu = p['alpha_mu']
    alpha_sigma = p['alpha_sigma']

    patient_size = pop_manager.get_patient_size()
    num_steps = int(x.size + x[0]/c.RESOLUTION)

    patients_alive = [patient_size] * num_steps

    initial_diameters = pop_manager.get_initial_diameters(
        stage_1=0,
        stage_2=c.RADIATION_ONLY_PATIENT_PERCENTAGE["2"],
        stage_3A=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3A"],
        stage_3B=c.RADIATION_ONLY_PATIENT_PERCENTAGE["3B"],
        stage_4=0)

    initial_volume =\
        pop_manager.get_volume_from_diameter(np.array(initial_diameters))

    alpha = np.array([alpha_mu, alpha_sigma, c.RAD_ALPHA[2], c.RAD_ALPHA[3]])
    rho = np.array(
        [rho_mu, rho_sigma, params['rho_mu'].min, params['rho_mu'].max])

    alpha_and_rho =\
        pop_manager.sample_correlated_params(alpha,
                                             rho,
                                             c.GR_RS_CORRELATION,
                                             retval=patient_size)

    treatment_delay = pop_manager.get_treatment_delay()
    treatment_days = pop_manager.get_radiation_days(treatment_delay, num_steps)
    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):

        obj_id =\
            sim_death_time_with_radiation.remote(num_steps,
                                                 initial_volume[num],
                                                 death_volume,
                                                 treatment_days[num],
                                                 func_pointer,
                                                 alpha_and_rho[num, 1],
                                                 K,
                                                 alpha=alpha_and_rho[num, 0],
                                                 beta=alpha_and_rho[num, 0]/10.
                                                 )

        id_list.append(obj_id)

    logging.info("Patient simulation complete, creating survival curve.")
    death_times = [ray.get(obj_id) for obj_id in id_list]

    for times in death_times:
        if times is not None:
            patients_alive = [(patients_alive[k] - 1) if x[k] >=
                              times * c.RESOLUTION else patients_alive[k] for k in range(num_steps)]
    patients_alive = np.array(patients_alive)
    patients_alive = (
        patients_alive/patients_alive[0])*(1 - c.SURVIVAL_REDUCTION/100.)
    patients_alive[0] = 1.
    patients_alive = patients_alive[int(
        x[0]/c.RESOLUTION):int(x[-1]/c.RESOLUTION)+1]
    months = x/31.

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return months, patients_alive
