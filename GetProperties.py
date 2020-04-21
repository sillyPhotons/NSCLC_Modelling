"""
Author: Ruiheng Su 2020

File containing definition of `PropertyManager` class.
"""

import csv
import logging
import numpy as np
import matplotlib as mpl
from lmfit import Parameters
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit

import Constants as c

class PropertyManager ():
    """
    Each `PropertyManager` object is associated with a integer `patient_size`.
    It contains function members that enables sampling normal and
    lognomral distributions and conversion methods between tumor parameters
    """

    def __init__(self, patient_size):
        """
        Params::
            `patient_size`: integer representing the number of patients in the Monte Carlo patient population

        Requires::
            `patient_size` is an integer

        Raises::
            Assertion error if argument `patient_size` is not an integer 
        """

        assert(isinstance(patient_size, int))

        self.patient_size = patient_size
        self.count = 0  # utility variable

    def sample_normal_param(self, mean, std, retval=1, lowerbound=None, upperbound=None):
        """
        Returns a numpy array or a python list of samples from a univariate 
        normal given `mean` and standard deviation (`std`) of the distribution.

        If the no upper and/or lower bound values are supplied, then a numpy 
        array with number of elements equal to `retval` is returned`.

        If either upper or lower bound value is supplied, then a python list is 
        returned

        Params::
            `mean`: mean of the normal distribution, scalar

            `std`: standard deviation of the normal distribution

            `retval`: integer number of values to return

            `upperbound`, `lowerbound`: upper and lower bound of returned values

        Requires::
            `upperbound > lowerbound`
            `!(mean == 0 and std == 0)`

        Raises::
            Assertion error if  `upperbound < lowerbound`
        """

        if (upperbound == None and lowerbound == None):

            return np.random.normal(mean, std, retval)

        elif (upperbound == None and lowerbound != None):

            data = list()
            i = 0
            while i < retval:
                point = np.random.normal(mean, std, 1)

                if (lowerbound < point):
                    data.append(point[0])
                    i += 1
            return data

        elif (upperbound != None and lowerbound == None):

            data = list()
            i = 0
            while i < retval:
                point = np.random.normal(mean, std, 1)

                if (point < upperbound):
                    data.append(point[0])
                    i += 1
            return data

        else:

            assert(upperbound > lowerbound)

            data = list()
            i = 0
            while i < retval:
                point = np.random.normal(mean, std, 1)

                if (lowerbound < point < upperbound):
                    data.append(point[0])
                    i += 1
            return data

    def sample_lognormal_param(self, mean, std, retval=1, lowerbound=None,
                               upperbound=None):
        """
        Returns a numpy array of samples from a univariate 
        log normal given `mean` and standard deviation (`std`) of the 
        underlying distribution.

        Params::
            `mean`: mean of the underlying normal distribution, scalar

            `std`: standard deviation of the underlying normal distribution

            `retval`: integer number of values to return

            `upperbound`, `lowerbound`: upper and lower bound of returned values

        Requires::
            `upperbound > lowerbound`
            `!(mean == 0 and std == 0)`

        Raises::
            Assertion error if  `upperbound < lowerbound`
        """

        if (upperbound == None and lowerbound == None):

            return np.random.lognormal(mean, std, retval)

        elif (upperbound == None and lowerbound != None):

            data = list()
            i = 0
            while i < retval:
                point = np.random.lognormal(mean, std, 1)

                if (lowerbound < point[0]):
                    data.append(point[0])
                    i += 1

            return np.array(data)

        elif (upperbound != None and lowerbound == None):

            data = list()
            i = 0
            while i < retval:
                point = np.random.lognormal(mean, std, 1)

                if (point[0] < upperbound):
                    data.append(point[0])
                    i += 1
            return np.array(data)

        else:

            assert(upperbound > lowerbound)
            lowerbound = (np.log(lowerbound) - mean) / std
            upperbound = (np.log(upperbound) - mean) / std

            norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=retval)

            data = np.exp((norm_rvs * std) + mean)

            return data

    def sample_correlated_params(self, param1, param2, corr, retval=1):
        """
        Given (mean, sigma, lowerbound, upperbound) of two values, and their 
        linear correlation coefficient, samples from a multivariate normal 
        distribution a number of ordered pairs equal to `retval`. The sampled 
        pairs are returned in a single numpy array with dimensions (2 by retval)

        Params::
            `param1`: numpy array with 2 elements. `param1[0]` is the mean of the parameter, `param1[1]` is the standard deviation, `param1[2]` is the lower bound this parameter, and `param1[3]` is the upper bound of this parameter

            `param2`: numpy array with 2 elements. `param2[0]` is the mean of the parameter, and `param2[1]` is the standard deviation, `param2[2]` is the lower bound this parameter, and `param2[3]` is the upper bound of this parameter

            `corr`: linear correlation coefficient of `param1` and `param2`. corr = (covariance of param1 and param2)/((sigma of param1)*(sigma of param 2))

            `retval`: number of ordered pairs to return 

        Requires::
            The upper and lower bound values are not `None`. Use `np.inf` instead

        Raises::
            Assertion error if any lower/upperbound values are `None`
        """
        sigma1 = param1[1]
        sigma2 = param2[1]
        lb1, ub1 = param1[2], param1[3]
        lb2, ub2 = param2[2], param2[3]

        assert(lb1 is not None)
        assert(ub1 is not None)
        assert(lb2 is not None)
        assert(ub2 is not None)

        covariance_matrix = np.array([[sigma1**2, corr * sigma1 * sigma2],
                                      [corr * sigma1 * sigma2, sigma2**2]])

        mean_array = np.array([param1[0], param2[0]])

        sampled_params = list()

        while len(sampled_params) < retval:

            pairs = np.random.multivariate_normal(
                mean_array, covariance_matrix, retval)

            for num in range(pairs.shape[0]):

                if lb1 < pairs[num, 0] < ub1 and lb2 < pairs[num, 1] < ub2:
                    sampled_params.append(pairs[num, :])

        return np.array(sampled_params)

    def get_patient_size(self):

        return self.patient_size

    def get_volume_from_diameter(self, diameter_array):
        """
        Given a numpy array with elements representing tumor diameter [cm], 
        converts each element to tumor volume [cm^3] assuming spherical tumor  

        Params::
            `diameter_array`: numpy array
        """

        return 4. / 3. * np.pi * (diameter_array / 2.) ** 3

    def get_diameter_from_volume(self, volume_array):
        """
        Given a numpy array with elements representing tumor volume [cm^3], 
        converts each element to tumor diameter [cm] assuming spherical tumor  

        Params::
            `volume_array`: numpy array
        """

        return np.cbrt((volume_array / (4./3. * np.pi))) * 2.

    def get_tumor_cell_number_from_diameter(self, diameter_array):
        """
        Given a numpy array with elements representing tumor diameter [cm], 
        converts each element to tumor cell number assuming spherical tumor  

        Params::
            `diameter_array`: numpy array
        """

        volume_array = self.get_volume_from_diameter(diameter_array)
        cell_number_array = volume_array * c.TUMOR_DENSITY

        return cell_number_array

    def get_tumor_cell_number_from_volume(self, volume_array):
        """
        Given a numpy array with elements representing tumor volume [cm^3], 
        converts each element to tumor cell number assuming spherical tumor  

        Params::
            `volume_array`: numpy array
        """

        cell_number_array = volume_array * c.TUMOR_DENSITY

        return cell_number_array

    def get_volume_from_tumor_cell_number(self, cell_number_array):
        """
        Given a numpy array with elements representing tumor cell number, 
        converts each element to tumor volume [cm^3] assuming spherical tumor  

        Params::
            `cell_number_array`: numpy array
        """

        volume_array = cell_number_array / c.TUMOR_DENSITY

        return volume_array

    def get_diameter_from_tumor_cell_number(self, cell_number_array):
        """
        Given a numpy array with elements representing tumor cell number, 
        converts each element to tumor diameter [cm] assuming spherical tumor  

        Params::
            `cell_number_array`: numpy array
        """

        volume_array = self.get_volume_from_tumor_cell_number(
            cell_number_array)
        diameter_array = self.get_diameter_from_volume(volume_array)

        return diameter_array

    def get_param_object_for_no_treatment(self, stage="1"):
        """
        Returns a `Parameters` object for no treatment result reproduction 
        (given a string representing the AJCC stage of the Monte Carlo 
        population to be simulated)

        Params::
            `stage`: a string

        Requires::
            `stage` can only take values of "1", "2", "3A", "3B", or "4"

        Raises::
            Assertion error if `stage` is not one of "1", "2", "3A", "3B", "4"
        """

        assert(stage in c.TABLE2.keys())

        params = Parameters()

        mu = np.log(c.TABLE2[stage][1])
        sigma = np.sqrt(2*(np.abs(np.log(c.TABLE2[stage][0]) - mu)))

        params.add("rho_mu", value=c.RHO[0],
                   min=c.RHO[2], max=c.RHO[3], vary=False)
        params.add("rho_sigma", value=c.RHO[1],
                   min=c.RHO[2], max=c.RHO[3], vary=False)
        params.add('K', value=self.get_volume_from_diameter(c.K),
                   min=0, vary=False)
        params.add('V_mu',
                   value=mu,
                   vary=False,
                   min=c.TABLE3[stage][2],
                   max=c.TABLE3[stage][3])
        params.add('V_sigma',
                   value=sigma,
                   vary=False,
                   min=c.TABLE3[stage][2],
                   max=c.TABLE3[stage][3])

        return params

    def get_param_object_for_radiation(self):
        """
        Returns a `Parameters` object to reproduce radiotherapy only results
        """

        params = Parameters()

        params.add("rho_mu", value=c.RHO[0],
                   min=c.RHO[2], max=c.RHO[3], vary=False)
        params.add("rho_sigma", value=c.RHO[1],
                   min=c.RHO[2], max=c.RHO[3], vary=False)
        params.add('K', value=self.get_volume_from_diameter(c.K),
                   min=0, vary=False)
        params.add("alpha_mu",
                   value=c.RAD_ALPHA[0],
                   vary=False,
                   min=c.RAD_ALPHA[2],
                   max=c.RAD_ALPHA[3])
        params.add("alpha_sigma",
                   value=c.RAD_ALPHA[1],
                   vary=False,
                   min=c.RAD_ALPHA[2],
                   max=c.RAD_ALPHA[3])

        return params

    def get_initial_diameters(self, stage_1=1, stage_2=0, stage_3A=0, stage_3B=0, stage_4=0):
        """
        Returns a python list where each entry is a diameter [cm] value sampled 
        from a population with a mixed proportion of a patients in each AJCC 
        stage.

        Params::
            `stage_1`: scalar, representing the fraction of stage I patients in the simulated population 
            `stage_1`: scalar, representing the fraction of stage I patients in the simulated population
            `stage_1`: scalar, representing the fraction of stage I patients in the simulated population
            `stage_1`: scalar, representing the fraction of stage I patients in the simulated population
            `stage_1`: scalar, representing the fraction of stage I patients in the simulated population

        Requires::
            `stage_1 + stage_2 + stage_3A + stage_3B + stage_4 == 1` 

        Raises::
            `stage_1 + stage_2 + stage_3A + stage_3B + stage_4 != 1`
        """

        assert(stage_1 + stage_2 + stage_3A + stage_3B + stage_4 == 1)

        stage_1_num = int(np.ceil(self.patient_size * stage_1))
        stage_2_num = int(np.ceil(self.patient_size * stage_2))
        stage_3A_num = int(np.ceil(self.patient_size * stage_3A))
        stage_3B_num = int(np.ceil(self.patient_size * stage_3B))
        stage_4_num = int(np.ceil(self.patient_size * stage_4))
        sum_num = stage_1_num + stage_2_num + stage_3A_num + stage_3B_num + stage_4_num

        if (sum_num != self.patient_size):
            logging.warning(
                "Requested {}, {} initial tumor diameter will be returned instead".format(self.patient_size, sum_num))

        stage_keys = c.TABLE2.keys()
        stage_num_dict = dict(zip(
            stage_keys, [stage_1_num, stage_2_num, stage_3A_num, stage_3B_num, stage_4_num]))

        ret_array = []
        for stage in stage_keys:
            if (stage_num_dict[stage] != 0):
                V_mu = np.log(c.TABLE2[stage][1])
                V_sigma = np.sqrt(
                    2*(np.abs(np.log(c.TABLE2[stage][0]) - V_mu)))
                lb = c.TABLE3[stage][2]
                ub = c.TABLE3[stage][3]

                lowerbound = (np.log(lb) - V_mu) / V_sigma
                upperbound = (np.log(ub) - V_mu) / V_sigma

                norm_rvs = truncnorm.rvs(
                    lowerbound, upperbound, size=stage_num_dict[stage])
                initial_diameter = list(np.exp((norm_rvs * V_sigma) + V_mu))
                ret_array = ret_array + initial_diameter

        assert(len(ret_array) == sum_num)

        return ret_array

    def get_treatment_delay(self, low=c.DIAGNOSIS_DELAY_RANGE[0], high=c.DIAGNOSIS_DELAY_RANGE[1]):
        """
        """
        return np.random.uniform(low=low,
                                 high=high,
                                 size=self.patient_size)

    def get_radiation_days(self, treatment_delay, num_steps):
        """
        Returns a 2 dimensional numpy array, with dimensions equal to the 
        number of Monte Carlo patients times the number of time steps (Each 
        row represents one patient, each column represents one time step). Each
        element of the array is either 1 or 0, with 1 representing that 
        radiation is applied at that time step, and  0 representing no 
        radiation. The dose fraction for each day is given in `Constants.py` as 
        `RAD_DOSE`, and the total dose is given as `TOTAL_DOSE`.

        A delay is uniformly sampled between a closed interval given by the 
        python list `DIAGNOSIS_DELAY_RANGE` in `Constants.py`. The radiation is 
        then applied for 5 days a week at `RAD_DOSE` fraction per day, to a 
        total of `TOTAL_DOSE`.   

        Params::
            `num_steps`: number of time steps. The time step has the basic unit of [days]

        Requires::
            `num_steps` is sufficently large so that total dose of `TOTAL_DOSE` [Gy] is achieved

        Raises::
            Assertion error if `TOTAL_DOSE` requirement is not met
        """

        treatment_days = np.zeros([self.patient_size, num_steps])

        one_day = int(1/c.RESOLUTION)

        days_with_rad = c.SCHEME[0] * one_day
        rest = c.SCHEME[1] * one_day

        fraction_per_step = c.RAD_DOSE/one_day
        for i in range(self.patient_size):
            steps_delayed = int(np.round(treatment_delay[i]/c.RESOLUTION))

            total_dose = 0
            last_step = steps_delayed

            while total_dose < c.TOTAL_DOSE:
                rad_days = last_step + days_with_rad
                treatment_days[i][last_step:rad_days] = 1
                total_dose += (fraction_per_step) * (rad_days - last_step)
                last_step = rad_days + rest

            entries = np.count_nonzero(treatment_days[i])

            assert(entries*fraction_per_step == c.TOTAL_DOSE)

        return treatment_days
