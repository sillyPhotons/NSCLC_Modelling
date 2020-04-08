"""
    This file contains two versions of the cost function. 
    
    `cost_function_no_treatment` involves the conversion from variables 
    associated from tumor diameter to tumor cell number, and back to tumor
    diameter to evaluate the death condition (diameter >= 13 cm)

    `cost_function_no_treatment_diameter` uses the variables associated with 
    tumor diameter as is, and solves the gompertz equation for diameter(time) as
    opposed to cell_number(time) in the aforemention cost function
"""
import logging
import numpy as np
from lmfit import minimize, Parameters
from scipy.integrate import odeint
from Model import gompertz_ode, gompertz_analytical, discrete_time_tumor_volume
import matplotlib.pyplot as plt
from Constants import DEATH_DIAMETER, RESOLUTION
import time

"""
    Returns the residual of model and data, given a Parameters object, the xy 
    data for a KMSc, and a PropertyManager object.

    Requires: 
        Parameters object contains Parameter objects with the keys:
            mean_growth_rate
            std_growth_rate
            carrying_capacity
            mean_tumor_diameter
            std_tumor_diameter
        The KMSc data at t = 0 must equal 1
        x,y series are numpy arrays
"""


def cost_function_no_treatment(params, x, data, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    start = time.time()

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
    print("Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)


"""
    Returns the residual of model and data, given a Parameters object, the xy 
    data for a KMSc, and a PropertyManager object.

    Requires: 
        Parameters object contains Parameter objects with the keys:
            mean_growth_rate
            std_growth_rate
            carrying_capacity
            mean_tumor_diameter
            std_tumor_diameter
        The KMSc data at t = 0 must equal 1
        x,y series are numpy arrays
"""


def cost_function_no_treatment_diameter(params, x, data, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    start = time.time()

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        solved_diameter = gompertz_analytical(
            tumor_diameter, x, growth_rate, carrying_capacity)

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
    print("Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)


def cost_function_no_treatment_volume(params, x, data, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    patient_size = pop_manager.get_patient_size()
    patients_alive = [patient_size] * len(x)

    start = time.time()

    for num in range(patient_size):

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=1, lowerbound=0.3, upperbound=5)[0]
        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=1, lowerbound=0, upperbound=None)[0]

        cell_volume = pop_manager.get_volume_from_diameter(
            tumor_diameter)

        solved_volume = gompertz_analytical(
            cell_volume, x, growth_rate, carrying_capacity)

        # odeint(gompertz_ode, cell_number, x, args=(
        #     growth_rate, carrying_capacity))

        solved_diameter = pop_manager.get_diameter_from_tumor_cell_number(
            solved_volume)

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
    print("Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)


def cost_function_discrete_time_volume(params, x, data, pop_manager):
    """
    `x`: corresponds to months

    Requires: 
        Difference between consecutive elements of data must be > 0.1 
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

    initial_diameter = pop_manager.sample_lognormal_param(
        mean=mean_tumor_diameter, std=std_tumor_diameter, retval=patient_size, lowerbound=params['mean_tumor_diameter'].min, upperbound=params['mean_tumor_diameter'].max)

    initial_volume = pop_manager.get_volume_from_diameter(np.array(initial_diameter))

    growth_rates = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=patient_size, lowerbound=0, upperbound=None)

    t = np.arange(x[0], x[-1], RESOLUTION)
    num_steps = t.size
    
    death_volume = pop_manager.get_volume_from_diameter(DEATH_DIAMETER) 
    cancer_volume = np.zeros((patient_size, num_steps))

    for num in range(patient_size):
        
        if (num % 100 == 0):
            logging.info("Simulating Patient {}".format(num))
            
        cancer_volume[num, 0] = initial_volume[num]
      
        death_time = None

        for i in range(1, num_steps):

            cancer_volume[num, i] = discrete_time_tumor_volume(cancer_volume[num, i - 1], growth_rates[num], carrying_capacity)

            if cancer_volume[num, i] > death_volume:
                cancer_volume[num, i] = death_volume
                death_time = i
                
                break 
    
        # plt.plot(x, solved_diameter)
        # plt.show()
        
        if (death_time is not None):
            death_time = t[death_time]
            patients_alive = [(patients_alive[k] - 1) if k >=
                              death_time else patients_alive[k] for k in range(len(x))]

    patients_alive = np.array(patients_alive)
    patients_alive = patients_alive/patients_alive[0]

    end = time.time()
    runtime = end - start

    logging.info("Minimization Iteration completed in {} seconds.".format(runtime))

    return (patients_alive - data)
