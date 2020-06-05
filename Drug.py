"""
Author: Ruiheng Su 2020

class defintion for a `Drug` object
"""

import numpy as np

class Drug ():
    """
    """
    def __init__ (self, days_array, dose, name):
        """
        """
        self.chemo_days = days_array
        self.dose = dose
        self.name = name

    @classmethod
    def from_days_array(cls, days_array, dose, halflife, name):
        """
        """
        return cls(days_array, dose, name)

    @classmethod
    def from_week_days_per_week(cls, week_day_array, num_weeks, offset, dose, halflife, name):
        """
        """
        for day in week_day_array:
            assert(day in range(1,8))

        assert len(week_day_array) <= 7

        offset_array = np.zeros(offset)
        week_array = [0 if i not in week_day_array else dose for i in range(1, 8)]*num_weeks
        week_array = np.array(week_array)
        days_array = np.concatenate([offset_array, week_array])
        return cls(days_array, dose, name)
