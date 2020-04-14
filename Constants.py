"""
File containing variables that are meant to be of constant value

Requires: The variables are never reassigned
"""


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

# Number of months passed per time step
RESOLUTION = 1

# 1.48% survival reduction
SURVIVAL_REDUCTION = 0

# 2 gray dose fractions
RAD_DOSE = 2

# Radiosensitivity parameter. alpha/beta = 10 (Mehta 2001)
RAD_ALPHA = [0.0398, 0.168]


TEST = {'1': [1.72, 4.70, 0.3, 5.0],
        '2': [1.96, 10.88, 0.3, 13.0],
        '3A': [1.91, 0.62, 0.3, 13.0],
        '3B': [2.76, 0.66, 0.3, 13.0],
        '4': [3.86, 0.89, 0.3, 13.0]}
