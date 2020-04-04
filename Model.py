"""
    This file contains the implementation of equations and models in Geng's 
    paper, which are used by the Main file, or the CostFunction file 
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from ReadData import read_file
from Constants import DEATH_DIAMETER
import time

"""
    Evaluates the right hand side of Equation 1. 

    N: A double value scalar
    t: numpy array representing time
    growth_rate: a scalar
    carrying_capacity: a scalar
"""


def gompertz_ode(N, t, growth_rate, carrying_capacity):

    dNdt = growth_rate*N*np.log(carrying_capacity/N)

    return dNdt


"""
    Returns a numpy array of gompertz equation evaluated at time = elements of provided parameter t.

    N0: Initial value at time = t[0]
    t: numpy array representing time
    growth_rate: scalar
    carrying_capacity: double scalar
"""


def gompertz_analytical(N0, t, growth_rate, carrying_capacity):
    # t = np.array(t)
    N = carrying_capacity * \
        np.exp(np.log(N0/carrying_capacity)*np.exp(-1.*growth_rate*t))
    return N


"""
    Given t and N as numpy arrays, computes the volume doubling time of a
    patient using equation (2) in Geng's paper.

    Requires: t must contain an element equal to 12, correponding to 12 months
              t and N are numpy arrays
"""
def volume_doubling_time(t, N):

    index = np.where(t == 12.0)[0]

    return (365)*np.log(2) / np.log(N[index]/N[0])


"""
    Returns a list of volume doubling time evaluated for each patient generated
    by the parameters specified by `params`. Can be used to plot the histogram 
    of volume doubling times (VDTs)

    params: Parameters object which must contain the following names as 
    Parameter objects
        mean_growth_rate
        std_growth_rate
        carrying_capacity
        mean_tumor_diameter
        std_tumor_diameter
    x: numpy array representing time
    pop_manager: PropertyManager object
"""
def predict_volume_doubling_time(params, x, pop_manager):

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
