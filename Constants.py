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
                         '4': [12, 8.82, 0.3, 13.0]}

DATA_PATIENT_SIZE = {'1': 1432,
                     "2": 128,
                     "3A": 1306,
                     "3B": 7248,
                     "4": 12840}

# Number of months passed per time step
RESOLUTION = 0.01

