"""
Author: Ruiheng Su 2020

File containing variables that are meant to be constants or given values in 
Geng's paper
"""

import numpy as np
# From table 3
TUMOR_SIZE_DISTRUTION = {'1': [2.50, 2.50, 0.3, 5.0],
                         '2': [3.50, 3.00, 0.3, None],
                         '3A': [6.60, 3.00, None],
                         '3B': [6.60, 3.00, 0.3, None]}

TUMOR_DENSITY = 5.8*10**8

DEATH_DIAMETER = 13.

REFER_TUMOR_SIZE_DIST = {'1': [1.72, 4.70, 0.3, 5.0],
                         '2': [1.96, 1.63, 0.3, 13.0],
                         '3A': [1.91, 9.40, 0.3, 13.0],
                         '3B': [2.76, 6.87, 0.3, 13.0],
                         '4': [3.86, 8.82, 0.3, 13.0]}

NATURAL_HISTORY_PATIENT_SIZE = {'1': 1432,
                                "2": 128,
                                "3A": 1306,
                                "3B": 7248,
                                "4": 12840}

# Gompertz growth rate
RHO = [7*10**-5, 7.23*10**-3, 0, np.inf]

# Gompertz carrying capacity(volume)
K = 30
"""
according tp ref # 27, the radiation therapy only patient group consisted of 6% 
stage II, 44% stage 3A, and 50% stage 3B
"""
RADIATION_ONLY_PATIENT_PERCENTAGE = {'2': 6/100.,
                                     '3A': 44/100.,
                                     '3B': 50/100.}

"""
The delay time, which is the time between diagnosis and the start of the
treatment, was uniformly sampled from 2 - 3 weeks
"""
DIAGNOSIS_DELAY_RANGE = [14, 21]

# Linear correlation coefficent between tumor growth rate and radiosensitivity
GR_RS_CORRELATION = 0.87
# GR_RS_CORRELATION = 0

# Number of months passed per time step
RESOLUTION = 1

# 1.48% survival reduction
SURVIVAL_REDUCTION = 0

# 2 gray dose fractions
RAD_DOSE = 2

# 60 Gy total radiation dose, 5 days a week at 2 Gy fractions
TOTAL_DOSE = 60

# Radiosensitivity parameter with units [Gy^-1]. alpha/beta = 10 (Mehta 2001)
# RAD_ALPHA = [0.0398, 0.168, 0, np.inf]
RAD_ALPHA = [0.16, 0.004, 0, np.inf]

# Values given in table 2 of the mean and median of the fitted volume distribution. To convert to input parameters for lognormal distribution sampling, mu = ln(mean), sigma = sqrt(2(ln(mean) - mu))
TABLE2 = {'1': [1.66, 1.23],
          '2': [4.49, 3.53],
          '3A': [5.63, 5.06],
          '3B': [8.54, 8.74],
          '4': [9.26, 9.68]}
