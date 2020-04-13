"""
    This file contains the implementation of equations and models in Geng's 
    paper, which are used by the Main file, or the CostFunction file 
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from ReadData import read_file
from Constants import DEATH_DIAMETER, RESOLUTION, RAD_DOSE
import time

# def radiation_dose (dose_step = False):
#     """
#     Returns a value equal to the radiation dose if `dose_step` is true
#     """

#     if dose_step:
#         return 2 
#     else:
#         return 0

def discrete_time_tumor_volume_GENG(previous_volume, growth_rate, carrying_capacity, rad_alpha=0, rad_beta=0, dose_step = False, h=RESOLUTION, noise=0):
    """
    Discrete time formulation of tumor volume function as seen in: 
    https://www.nature.com/articles/s41598-018-30761-7
    """
    dose = 0
    if dose_step:
        dose = RAD_DOSE

    return previous_volume * np.exp(growth_rate * h * np.log(carrying_capacity/previous_volume) - (rad_alpha*dose + rad_beta*dose**2))


def rk4_tumor_volume(previous_volume, growth_rate, carrying_capacity, rad_alpha=0, rad_beta=0, dose_step = False, h=RESOLUTION, noise=0):
    """
    """

    dose = 0
    if dose_step:
        dose = RAD_DOSE

    k1 = previous_volume * growth_rate * \
        np.log(carrying_capacity / previous_volume) - (rad_alpha*dose + rad_beta*dose**2)* previous_volume

    k2 = (previous_volume + h * k1 / 2.) * growth_rate * \
        np.log(carrying_capacity / (previous_volume + h * k1 / 2.)) - (rad_alpha*dose + rad_beta*dose**2) * (previous_volume + h * k1 / 2.)

    k3 = (previous_volume + h * k2 / 2.) * growth_rate * \
        np.log(carrying_capacity / (previous_volume + h * k2 / 2.)) - (rad_alpha*dose + rad_beta*dose**2) * (previous_volume + h * k2 / 2.)

    k4 = (previous_volume + h * k3) * growth_rate * \
        np.log(carrying_capacity / (previous_volume + h * k3)) - (rad_alpha*dose + rad_beta*dose**2) * (previous_volume + h * k3)

    return previous_volume + (1./6) * h * (k1 + 2*k2 + 2*k3 + k4) 


def euler_tumor_volume(previous_volume, growth_rate, carrying_capacity, rad_alpha=0, rad_beta=0, dose_step = False, h=RESOLUTION, noise=0):
    """
    Discrete time formulation of tumor volume via linear approximation
    """
    dose = 0
    if dose_step:
        dose = RAD_DOSE

    return previous_volume + h * previous_volume * growth_rate * np.log(carrying_capacity / previous_volume) - (rad_alpha*dose + rad_beta*dose**2)*previous_volume


def gompertz_ode(N, t, growth_rate, carrying_capacity):
    """
    Evaluates the right hand side of Equation 1. 

    N: A double value scalar
    t: numpy array representing time
    growth_rate: a scalar
    carrying_capacity: a scalar
    """

    dNdt = growth_rate*N*np.log(carrying_capacity/N)

    return dNdt


def gompertz_analytical(N0, t, growth_rate, carrying_capacity):
    """
    Returns a numpy array of gompertz equation evaluated at time = elements of provided parameter t.

    N0: Initial value at time = t[0]
    t: numpy array representing time
    growth_rate: scalar
    carrying_capacity: double scalar
    """

    # t = np.array(t)
    N = carrying_capacity * \
        np.exp(np.log(N0/carrying_capacity)*np.exp(-1.*growth_rate*t))
    return N


def volume_doubling_time(V0, V1):
    """
    Given the initial tumor volume and volume after 1 year, calculate the volume doubling time
    """
    return (365)*np.log(2) / np.log(V1/V0)


def predict_volume_doubling_time(params, x, pop_manager):
    """
    Returns a list of volume doubling time evaluated for each patient generated
    by the parameters specified by `params`. Can be used to plot the histogram 
    of volume doubling times (VDTs)

    `params`: Parameters object which must contain the following names as 
    Parameter objects
        `mean_growth_rate`
        `std_growth_rate`
        `carrying_capacity`
        `mean_tumor_diameter`
        `std_tumor_diameter`
    `x`: numpy array representing time
    `pop_manager`: `PropertyManager` object
    """

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    VDT_hist = []

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]

        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        solved_diameter = odeint(gompertz_ode, tumor_diameter, x, args=(
            growth_rate, carrying_capacity))

        VDT_hist.append(volume_doubling_time(x, solved_diameter)[0][0])

    return VDT_hist


def predict_no_treatment_diameter(params, x, pop_manager):
    """
    Given a fixed set of parameters, an array representing time for which this 
    model is evaluation on, and a PropertyManager object, returns the time array
    and the result array of no treatment received model

    params: Parameters object which must contain Parameter objects with the 
    following names
        mean_growth_rate
        std_growth_rate
        carrying_capacity
        mean_tumor_diameter
        std_tumor_diameter
    x: numpy array representing time
    pop_manager: PropertyManager object
    """
    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)
    start = time.time()
    print("Began model evaluation")

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        solved_diameter = odeint(gompertz_ode, tumor_diameter, x, args=(
            growth_rate, carrying_capacity))

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

    end = time.time()
    runtime = end - start
    print("Predictive model evaluated in {} seconds.".format(runtime))

    return x, patients_alive


def predict_no_treatment(params, x, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)
    start = time.time()
    print("Began model evaluation")

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        cell_number = pop_manager.get_tumor_cell_number_from_diameter(
            tumor_diameter)

        solved_cell_number = gompertz_analytical(
            cell_number, x, growth_rate, carrying_capacity)

        # odeint(gompertz_ode, cell_number, x, args=(
        #     growth_rate, carrying_capacity))

        solved_diameter = pop_manager.get_diameter_from_tumor_cell_number(
            solved_cell_number)

        # plt.plot(x, solved_diameter)
        # plt.show()

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
    end = time.time()
    runtime = end - start
    print("Predictive model evaluated in {} seconds.".format(runtime))

    return x, patients_alive


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

    initial_volume = pop_manager.get_volume_from_diameter(
        np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
        mean=mean_growth_rate, std=std_growth_rate, retval=patient_size, lowerbound=0, upperbound=None)

    num_steps = x.size

    death_volume = pop_manager.get_volume_from_diameter(DEATH_DIAMETER)
    cancer_volume = np.zeros((patient_size, num_steps))

    pop_manager.count = 0
    for num in range(patient_size):

        if (num % 100 == 0 and num != 0):
            logging.info(
                "\U0001F637 {} Patients Died".format(pop_manager.count))
            logging.info("Simulating Patient {}".format(num))
            pop_manager.count = 0

        cancer_volume[num, 0] = initial_volume[num]

        death_time = None
        for i in range(1, num_steps):

            cancer_volume[num, i] = func_pointer(
                cancer_volume[num, i - 1], growth_rates[num], carrying_capacity, h=RESOLUTION)

            if cancer_volume[num, i] > death_volume:
                cancer_volume[num, i] = death_volume
                death_time = i
                pop_manager.count += 1
                break

        if (death_time is not None):
            patients_alive = [(patients_alive[k] - 1) if k >=
                              death_time else patients_alive[k] for k in range(num_steps)]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    end = time.time()
    runtime = end - start

    logging.info(
        "Minimization Iteration completed in {} seconds.".format(runtime))

    months = [num / 31. for num in x]

    return months, patients_alive
