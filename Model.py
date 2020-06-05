"""
Author: Ruiheng Su 2020

File containing implementations of discrete time models of Equation (7) in 
Geng's paper, and a function to calculate volume doubling time 
"""
import time
import logging
import numpy as np
import Constants as c
from ReadData import read_file
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters

def tumor_volume_GENG_Logistic(V0, rho, K, alpha=0, beta=0, dose_step=0, noise=0):
    """
    Modification to the iscrete time formulation of tumor volume function as 
    seen in: https://www.nature.com/articles/s41598-018-30761-7. Uses the 
    logistic model for tumor growth instead of gompertz model. A different 
    optimized value for rho is required for this model.

    Returns tumor volume one `c.RESOLUTION` time step ahead.
    
    Params::
        `V0`: scalar value representing previous tumor volume

        `rho`: scalar value representing gompertz growth rate 

        `alpha`: radiosensitivity parameter [Gy^-1] 

        `beta`:

        `dose_step`: 0, or 1, represents whether this time step involves the application of radiation

        `noise`: noise to be added

    Requires::
        `V0 != 0` 
        `K != 0`
        All arguments are scalars

    Raises::
        Assertion error if `V0 == 0`
    """

    assert(V0 != 0)

    dose = 0
    if dose_step:
        dose = c.RAD_DOSE*c.RESOLUTION

    return V0 * np.exp(c.RESOLUTION * rho * V0 *(1 - V0/K)\
         - (alpha*dose + beta*dose**2)) + noise

def tumor_volume_GENG(V0, rho, K, alpha=0, beta=0, dose_step=0,\
     noise=0):
    """
    Discrete time formulation of tumor volume function as seen in: 
    https://www.nature.com/articles/s41598-018-30761-7

    Returns tumor volume one `c.RESOLUTION` time step ahead.
    
    Params::
        `V0`: scalar value representing previous tumor volume

        `rho`: scalar value representing gompertz growth rate 

        `alpha`: radiosensitivity parameter [Gy^-1] 

        `beta`:

        `dose_step`: 0, or 1, represents whether this time step involves the application of radiation

        `noise`: noise to be added

    Requires::
        `V0 != 0` 
        `K != 0`
        All arguments are scalars

    Raises::
        Assertion error if `V0 == 0`
    """

    assert(V0 != 0)

    dose = 0
    if dose_step:
        dose = c.RAD_DOSE*c.RESOLUTION

    return V0 * np.exp(c.RESOLUTION * rho * np.log(K/V0)\
         - (alpha*dose + beta*dose**2)) + noise

         

#FIXME
def rk4_tumor_volume(V0, rho, K, alpha=0, beta=0, dose_step=0, noise=0):
    """
    4th Order Runge Kutta method to solve the ODE (Equation 7)

    Returns tumor volume one `c.RESOLUTION` time step ahead.
    
    Params::
        `V0`: scalar value representing previous tumor volume

        `rho`: scalar value representing gompertz growth rate 

        `alpha`: radiosensitivity parameter [Gy^-1] 

        `beta`:

        `dose_step`: 0, or 1, represents whether this time step involves the application of radiation

        `noise`: noise to be added

    Requires::
        `V0 != 0` 
        `K != 0`
        All arguments are scalars

    Raises::
        Assertion error if `V0 == 0`
    """

    assert(V0 != 0)

    dose = 0
    if dose_step:
        dose = c.RAD_DOSE*c.RESOLUTION

    k1 = V0 * rho * \
        np.log(K / V0) - (alpha *
                          dose + beta*dose**2) * V0

    k2 = (V0 + (c.RESOLUTION * k1 / 2.)) * rho * \
        np.log(K / (V0 + (c.RESOLUTION * k1 / 2.))) \
             - (alpha * dose + beta*dose**2) * (V0 + (c.RESOLUTION * k1 / 2.))

    k3 = (V0 + (c.RESOLUTION * k2 / 2.)) * rho * \
        np.log(K / (V0 + (c.RESOLUTION * k2 / 2.))) \
             - (alpha * dose + beta*dose**2) * (V0 + (c.RESOLUTION * k2 / 2.))

    k4 = (V0 + (c.RESOLUTION * k3)) * rho * \
        np.log(K / (V0 + (c.RESOLUTION * k3))) \
            - (alpha * dose + beta*dose**2) * (V0 + (c.RESOLUTION * k3))

    return V0 + (1./6.) * c.RESOLUTION * (k1 + 2*k2 + 2*k3 + k4) + noise


def euler_tumor_volume(V0, rho, K, alpha=0, beta=0, dose_step=False, noise=0):
    """
    Euler's method to solve the ODE (Equation 7)

    Return tumor volume one `c.RESOLUTION` time step ahead.

    Returns tumor volume one `c.RESOLUTION` time step ahead.
    
    Params::
        `V0`: scalar value representing previous tumor volume

        `rho`: scalar value representing gompertz growth rate 

        `alpha`: radiosensitivity parameter [Gy^-1] 

        `beta`:

        `dose_step`: 0, or 1, represents whether this time step involves the application of radiation

        `noise`: noise to be added
        
    Requires::
        `V0 != 0` 
        `K != 0`
        All arguments are scalars

    Raises::
        Assertion error if `V0 == 0`
    """

    assert(V0 != 0)

    dose = 0
    if dose_step:
        dose = c.RAD_DOSE*c.RESOLUTION

    return V0 + c.RESOLUTION * V0 * rho * np.log(K / V0)\
         - (alpha*dose + beta*dose**2)*V0 + noise


def volume_doubling_time(V0, V1):
    """
    Given the initial tumor volume and volume after 1 year, returns the volume doubling time [days]

    Params::
        `V0`: initial tumor volume as a scalar
        
        `V1`: tumor volume after 365 days as a scalar

    Requires::
        `V0 != 0`
        `V1 != 0`
        `V1 != V0`

    Raises::
        Assertion error if requirements are violated
    """
    assert(V0 != 0 and V1 != 0)
    assert(V0 != V1)

    return (365)*np.log(2) / np.log(V1/V0)

def drug_concentration (C_max, half_life, time):

    return C_max * np.exp((-np.log(2) / half_life)*time)

# def gompertz_ode(N, t, growth_rate, K):
#     """
#     Evaluates the right hand side of Equation 1.

#     N: A double value scalar
#     t: numpy array representing time
#     growth_rate: a scalar
#     K: a scalar
#     """

#     dNdt = growth_rate*N*np.log(K/N)

#     return dNdt


# def gompertz_analytical(N0, t, growth_rate, K):
#     """
#     Returns a numpy array of gompertz equation evaluated at time = elements of provided parameter t.

#     N0: Initial value at time = t[0]
#     t: numpy array representing time
#     growth_rate: scalar
#     K: double scalar
#     """

#     N = K * \
#         np.exp(np.log(N0/K)*np.exp(-1.*growth_rate*t))
#     return N


# def predict_volume_doubling_time(params, x, pop_manager):
#     """
#     Returns a list of volume doubling time evaluated for each patient generated
#     by the parameters specified by `params`. Can be used to plot the histogram
#     of volume doubling times (VDTs)

#     `params`: Parameters object which must contain the following names as
#     Parameter objects
#         `rho_mu`
#         `rho_sigma`
#         `K`
#         `V_mu`
#         `V_sigma`
#     `x`: numpy array representing time
#     `pop_manager`: `PropertyManager` object
#     """

#     p = params.valuesdict()
#     rho_mu = p['rho_mu']
#     rho_sigma = p['rho_sigma']
#     K = p['K']
#     V_mu = p['V_mu']
#     V_sigma = p['V_sigma']

#     patient_size = pop_manager.get_patient_size()
#     VDT_hist = []

#     for num in range(patient_size):

#         tumor_diameter = pop_manager.sample_lognormal_param(
#             mean=V_mu, std=V_sigma, retval=1, lowerbound=0.3, upperbound=5)[0]

#         growth_rate = pop_manager.sample_normal_param(
#             mean=rho_mu, std=rho_sigma, retval=1, lowerbound=0, upperbound=None)[0]

#         solved_diameter = odeint(gompertz_ode, tumor_diameter, x, args=(
#             growth_rate, K))

#         VDT_hist.append(volume_doubling_time(x, solved_diameter)[0][0])

#     return VDT_hist


# def predict_no_treatment_diameter(params, x, pop_manager):
#     """
#     Given a fixed set of parameters, an array representing time for which this
#     model is evaluation on, and a PropertyManager object, returns the time array
#     and the result array of no treatment received model

#     params: Parameters object which must contain Parameter objects with the
#     following names
#         rho_mu
#         rho_sigma
#         K
#         V_mu
#         V_sigma
#     x: numpy array representing time
#     pop_manager: PropertyManager object
#     """
#     p = params.valuesdict()
#     rho_mu = p['rho_mu']
#     rho_sigma = p['rho_sigma']
#     K = p['K']
#     V_mu = p['V_mu']
#     V_sigma = p['V_sigma']

#     patient_size = pop_manager.get_patient_size()
#     patients_alive = [patient_size] * len(x)
#     start = time.time()
#     print("Began model evaluation")

#     for num in range(patient_size):

#         tumor_diameter = pop_manager.sample_lognormal_param(
#             mean=V_mu, std=V_sigma, retval=1, lowerbound=0.3, upperbound=5)[0]
#         growth_rate = pop_manager.sample_normal_param(
#             mean=rho_mu, std=rho_sigma, retval=1, lowerbound=0, upperbound=None)[0]

#         solved_diameter = odeint(gompertz_ode, tumor_diameter, x, args=(
#             growth_rate, K))

#         try:
#             death_time = next(x for x, val in enumerate(solved_diameter)
#                               if val >= DEATH_DIAMETER)

#         except:
#             death_time = None

#         if (death_time is not None):
#             patients_alive = [(patients_alive[num] - 1) if num >=
#                               death_time else patients_alive[num] for num in range(len(x))]

#     patients_alive = np.array(patients_alive)
#     patients_alive = patients_alive/patients_alive[0]

#     end = time.time()
#     runtime = end - start
#     print("Predictive model evaluated in {} seconds.".format(runtime))

#     return x, patients_alive


# def predict_no_treatment(params, x, pop_manager):

#     p = params.valuesdict()
#     rho_mu = p['rho_mu']
#     rho_sigma = p['rho_sigma']
#     K = p['K']
#     V_mu = p['V_mu']
#     V_sigma = p['V_sigma']

#     patient_size = pop_manager.get_patient_size()
#     patients_alive = [patient_size] * len(x)
#     start = time.time()
#     print("Began model evaluation")

#     for num in range(patient_size):

#         tumor_diameter = pop_manager.sample_lognormal_param(
#             mean=V_mu, std=V_sigma, retval=1, lowerbound=0.3, upperbound=5)[0]
#         growth_rate = pop_manager.sample_normal_param(
#             mean=rho_mu, std=rho_sigma, retval=1, lowerbound=0, upperbound=None)[0]

#         cell_number = pop_manager.get_tumor_cell_number_from_diameter(
#             tumor_diameter)

#         solved_cell_number = gompertz_analytical(
#             cell_number, x, growth_rate, K)

#         # odeint(gompertz_ode, cell_number, x, args=(
#         #     growth_rate, K))

#         solved_diameter = pop_manager.get_diameter_from_tumor_cell_number(
#             solved_cell_number)

#         # plt.plot(x, solved_diameter)
#         # plt.show()

#         try:
#             death_time = next(x for x, val in enumerate(solved_diameter)
#                               if val >= DEATH_DIAMETER)

#         except:
#             death_time = None

#         if (death_time is not None):
#             patients_alive = [(patients_alive[num] - 1) if num >=
#                               death_time else patients_alive[num] for num in range(len(x))]

#     patients_alive = np.array(patients_alive)
#     patients_alive = patients_alive/patients_alive[0]
#     end = time.time()
#     runtime = end - start
#     print("Predictive model evaluated in {} seconds.".format(runtime))

#     return x, patients_alive


# def predict_KMSC_discrete(params, x, pop_manager, func_pointer):
#     """
#     """
#     start = time.time()

#     from scipy.stats import truncnorm

#     p = params.valuesdict()
#     rho_mu = p['rho_mu']
#     rho_sigma = p['rho_sigma']
#     K = p['K']
#     V_mu = p['V_mu']
#     V_sigma = p['V_sigma']

#     patient_size = pop_manager.get_patient_size()
#     patients_alive = [patient_size] * len(x)

#     ######################################################################
#     lowerbound = (np.log(params['V_mu'].min) -
#                   V_mu) / V_sigma
#     upperbound = (np.log(params['V_mu'].max) -
#                   V_mu) / V_sigma

#     norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=patient_size)

#     initial_diameter = list(np.exp(
#         (norm_rvs * V_sigma) + V_mu))
#     ######################################################################

#     # ######################################################################
#     # lowerbound = params['V_mu'].min
#     # upperbound = params['V_mu'].max

#     # initial_diameter = pop_manager.sample_lognormal_param(V_mu,
#     #                                                       V_sigma,
#     #                                                       retval=patient_size,
#     #                                                       lowerbound=lowerbound,
#     #                                                       upperbound=upperbound)
#     # ######################################################################

#     initial_volume = pop_manager.get_volume_from_diameter(
#         np.array(initial_diameter))

#     growth_rates = pop_manager.sample_normal_param(
#         mean=rho_mu, std=rho_sigma, retval=patient_size, lowerbound=0, upperbound=None)

#     num_steps = x.size

#     death_volume = pop_manager.get_volume_from_diameter(c.DEATH_DIAMETER)
#     cancer_volume = np.zeros((patient_size, num_steps))

#     pop_manager.count = 0
#     for num in range(patient_size):

#         if (num % 100 == 0 and num != 0):
#             logging.info(
#                 "\U0001F637 {} Patients Died".format(pop_manager.count))
#             logging.info("Simulating Patient {}".format(num))
#             pop_manager.count = 0

#         cancer_volume[num, 0] = initial_volume[num]

#         death_time = None
#         for i in range(1, num_steps):

#             cancer_volume[num, i] = func_pointer(
#                 cancer_volume[num, i - 1], growth_rates[num], K, h=c.RESOLUTION)

#             if cancer_volume[num, i] > death_volume:
#                 cancer_volume[num, i] = death_volume
#                 death_time = i
#                 pop_manager.count += 1
#                 break

#         if (death_time is not None):
#             patients_alive = [(patients_alive[k] - 1) if k >=
#                               death_time else patients_alive[k] for k in range(num_steps)]

#     patients_alive = np.array(patients_alive)
#     patients_alive = patients_alive/patients_alive[0]

#     end = time.time()
#     runtime = end - start

#     logging.info(
#         "Minimization Iteration completed in {} seconds.".format(runtime))

#     months = [num / 31. for num in x]

#     return months, patients_alive
