"""
Author: Ruiheng Su 2020

File containing variables that are meant to be constants or given values in 
Geng's paper
"""

import numpy as np

# 13 cm tumor diameter corresponds to patient death
DEATH_DIAMETER = 13.

# [N/cm^3] Number of cells per 1 cm^3 volume of tumor
TUMOR_DENSITY = 5.8*10**8

# From table 3, (mean, sigma, lower bound, upper bound)
TABLE3 = {'1': [1.72, 4.70, 0.3, 5.0],
          '2': [1.96, 1.63, 0.3, 13.0],
          '3A': [1.91, 9.40, 0.3, 13.0],
          '3B': [2.76, 6.87, 0.3, 13.0],
          '4': [3.86, 8.82, 0.3, 13.0]}

# Detterbeck 2008
NATURAL_HISTORY_PATIENT_SIZE = {'1': 1432,
                                "2": 128,
                                "3A": 1306,
                                "3B": 7248,
                                "4": 12840}

# Gompertz carrying capacity given as diameter (30 cm)
K = 30

# Gompertz growth rate. (mean, sigma, lower bound, upper bound)
RHO = [7*10**-5, 7.23*10**-3, 0, np.inf]


"""
according tp ref # 27, the radiation therapy only patient group consisted of 6% 
stage II, 44% stage 3A, and 50% stage 3B
"""
RADIATION_ONLY_PATIENT_PERCENTAGE = {'1': 0,
                                     '2': 6/100.,
                                     '3A': 44/100.,
                                     '3B': 50/100.,
                                     "4": 0}


"""
The delay time, which is the time between diagnosis and the start of the
treatment, was uniformly sampled from 2 - 3 weeks
"""
DIAGNOSIS_DELAY_RANGE = [14, 21]
DIAGNOSIS_DELAY_RANGE = [0,0]

# Linear correlation coefficent between tumor growth rate and radiosensitivity
# GR_RS_CORRELATION = 0.87
GR_RS_CORRELATION = 0

# Number of months passed per time step
RESOLUTION = 0.25

# 1.48% survival reduction
SURVIVAL_REDUCTION = 0

# 2 gray dose fractions each day
# must be > 0
RAD_DOSE = 2

# 60 Gy total radiation dose, 5 days a week at 2 Gy fractions
TOTAL_DOSE = 0

# RTOG8808 Patients received 2 Gy fractions 5 days followed by 2 day rest
SCHEME = [5, 2]

# Radiosensitivity parameter with units [Gy^-1]. alpha/beta = 10 (Mehta 2001)
RAD_ALPHA = [0.0398, 0.168, 0, np.inf]

ALPHA_PER_BETA = 10.

"""
Values given in table 2 of the mean and median of the fitted volume 
distribution. To convert to input parameters for lognormal distribution 
sampling, mu = ln(mean), sigma = sqrt(2(ln(mean) - mu))
"""
TABLE2 = {'1': [1.66, 1.23],
          '2': [4.49, 3.53],
          '3A': [5.63, 5.06],
          '3B': [8.54, 8.74],
          '4': [9.26, 9.68]}
