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
import Constants as c
import Model as m


@ray.remote
def sim_patient_death_time(num_steps, initial_volume, death_volume, func_pointer, *func_args, **func_kwargs):
    """
    This function is decorated with `@ray.remote`, which means that it is a funciton that may be called multiple times in parallel. Given the parameters, returns a single integer equal to number of time steps taken before the patient tumor volume exceeds `death_volume`
    `num_steps`: number of `RESOLUTION` steps to take
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
def sim_patient_death_time_with_radiation(num_steps, initial_volume, death_volume, treatment_delay, func_pointer, *func_args, **func_kwargs):
    """
    This function is decorated with `@ray.remote`, which means that it is a funciton that may be called multiple times in parallel. Given the parameters, returns a single integer equal to number of time steps taken before the patient tumor volume exceeds `death_volume`
    `num_steps`: number of `RESOLUTION` steps to take
    `initial_volume`: the initial tumor volume
    `death_volume`: volume of tumor at which point the patient is considered dead
    `func_pointer`: discrete time model of the model taking `*func_args` and `**func_kwargs` as parameters
    """

    cancer_volume = np.zeros(num_steps)
    cancer_volume[0] = initial_volume
    recover_prob = np.random.rand(num_steps)
    delay_steps = int(treatment_delay/c.RESOLUTION)
    skip = int(1/c.RESOLUTION) - 1

    death_time = None
    total_dose = 0
    skipped = False
    for i in range(1, num_steps):

        if i > delay_steps and skipped == False:

            cancer_volume[i] = func_pointer(
                cancer_volume[i - 1], *func_args, **func_kwargs, dose_step = True)

            if cancer_volume[i] > death_volume:
                cancer_volume[i] = death_volume
                death_time = i
                break

            # probability that the tumor was controlled
            if recover_prob[i] < np.exp(-cancer_volume[i] * c.TUMOR_DENSITY):
                return None

            total_dose += 2

    if (death_time is not None):

        return death_time
    else:
        return None


def predict_KMSC_discrete(params, x, pop_manager, func_pointer):
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

    from scipy.stats import truncnorm

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    V_mu = p['V_mu']
    V_sigma = p['V_sigma']

    patient_size = pop_manager.get_patient_size()
    num_steps = x.size
    patients_alive = [patient_size] * num_steps

    ######################################################################
    lowerbound = (np.log(params['V_mu'].min) -
                  V_mu) / V_sigma
    upperbound = (np.log(params['V_mu'].max) -
                  V_mu) / V_sigma

    norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=patient_size)

    initial_diameter = list(np.exp(
        (norm_rvs * V_sigma) + V_mu))
    ######################################################################

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
        mean=rho_mu, std=rho_sigma, retval=patient_size, lowerbound=0, upperbound=None)

    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_death_time.remote(num_steps,
                                               initial_volume[num], death_volume, func_pointer, growth_rates[num], K)

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

    from scipy.stats import truncnorm

    p = params.valuesdict()
    rho_mu = p['rho_mu']
    rho_sigma = p['rho_sigma']
    K = p['K']
    V_mu = p['V_mu']
    V_sigma = p['V_sigma']

    patient_size = pop_manager.get_patient_size()

    ######################################################################
    lowerbound = (np.log(params['V_mu'].min) -
                  V_mu) / V_sigma
    upperbound = (np.log(params['V_mu'].max) -
                  V_mu) / V_sigma

    norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=patient_size)

    initial_diameter = list(np.exp(
        (norm_rvs * V_sigma) + V_mu))
    ######################################################################

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

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


def reproduce_KMSC_discrete_with_radiation(params, x, pop_manager, func_pointer):

    start = time.time()

    from scipy.stats import truncnorm

    patient_size = pop_manager.get_patient_size()
    num_steps = x.size
    patients_alive = [patient_size] * num_steps

    stage2_num = patient_size * c.RADIATION_ONLY_PATIENT_PERCENTAGE["2"]
    stage3A_num = patient_size * c.RADIATION_ONLY_PATIENT_PERCENTAGE["3A"]
    stage3B_num = patient_size * c.RADIATION_ONLY_PATIENT_PERCENTAGE["3B"]

    stage_pop = {"2": int(stage2_num),
                 "3A": int(stage3A_num),
                 "3B": int(stage3B_num)}

    diameters = []
    for stage in ["2", "3A", "3B"]:
        V_mu = c.REFER_TUMOR_SIZE_DIST[stage][0]
        V_sigma = c.REFER_TUMOR_SIZE_DIST[stage][1]

        ######################################################################
        lowerbound = (np.log(c.REFER_TUMOR_SIZE_DIST[stage][2]) -
                      V_mu) / V_sigma
        upperbound = (np.log(c.REFER_TUMOR_SIZE_DIST[stage][3]) -
                      V_mu) / V_sigma

        norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=stage_pop[stage])

        initial_diameter = list(np.exp(
            (norm_rvs * V_sigma) + V_mu))
        ######################################################################
        diameters.append(initial_diameter)

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(diameters))

    rad_alpha = np.array(c.RAD_ALPHA)
    rho = np.array([7.00*10**-5, 7.23*10**-3])

    rad_alpha_and_rho = pop_manager.sample_correlated_params(
        rad_alpha, rho, c.GR_RS_CORRELATION, retval=patient_size)

    treatment_delay = np.random.uniform(low=c.DIAGNOSIS_DELAY_RANGE[0],
                                        high=c.DIAGNOSIS_DELAY_RANGE[1],
                                        size=patient_size)

    death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)

    id_list = list()
    for num in range(patient_size):

        obj_id = sim_patient_death_time_with_radiation.remote(num_steps,
                                        initial_volume[num], death_volume, treatment_delay[num], func_pointer, rad_alpha_and_rho[num, 1], pop_manager.get_volume_from_diameter(30), rad_alpha=rad_alpha_and_rho[num, 0], rad_beta=rad_alpha_and_rho[num, 0]/10.)

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

    months = x/31.

    end = time.time()
    runtime = end - start

    logging.info(
        "\U0001F637 Minimization Iteration completed in {} seconds.".format(runtime))

    return months, patients_alive
