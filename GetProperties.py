import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from Constants import STAGE, TUMOR_DENSITY
import csv

"""
"""


class PropertyManager ():

    patient_size = None

    def __init__(self, patient_size, ):
        PropertyManager.patient_size = patient_size

    def sample_normal_param(self, mean, std, retval=1, upperbound=None, lowerbound=None):

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
            data = list()
            i = 0
            while i < retval:
                point = np.random.normal(mean, std, 1)

                if (lowerbound < point < upperbound):
                    data.append(point[0])
                    i += 1
            return data

    def sample_lognormal_param(self, mean, std, retval=1, lowerbound=None, upperbound=None):

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

            return data

        elif (upperbound != None and lowerbound == None):

            data = list()
            i = 0
            while i < retval:
                point = np.random.lognormal(mean, std, 1)

                if (point[0] < upperbound):
                    data.append(point[0])
                    i += 1
            return data

        else:
            data = list()
            i = 0
            while i < retval:
                point = np.random.lognormal(mean, std, 1)
                if (lowerbound < point[0] < upperbound):
                    data.append(point[0])
                    i += 1
            return data

    def get_patient_size(self):

        return PropertyManager.patient_size

    def get_volume_from_diameter(self, diameter_array):

        volume_array = (4.*np.pi/3.) * (diameter_array/2.)**2

        return volume_array

    def get_diameter_from_volume(self, volume_array):

        diameter_array = np.sqrt((3./(4*np.pi)) * volume_array) * 2

        return diameter_array

    def get_tumor_cell_number_from_diameter(self, diameter_array):

        volume_array = self.get_volume_from_diameter(diameter_array)
        cell_number_array = volume_array * TUMOR_DENSITY

        return cell_number_array

    def get_tumor_cell_number_from_volume(self, volume_array):

        cell_number_array = volume_array * TUMOR_DENSITY

        return cell_number_array

    def get_volume_from_tumor_cell_number(self, cell_number_array):

        volume_array = cell_number_array / TUMOR_DENSITY

        return volume_array

    def get_diameter_from_tumor_cell_number(self, cell_number_array):

        volume_array = self.get_volume_from_tumor_cell_number(
            cell_number_array)
        diameter_array = self.get_diameter_from_volume(volume_array)

        return diameter_array


def generate_csv(csv_path, params, pop_manager):

    p = params.valuesdict()
    mean_growth_rate = p['mean_growth_rate']
    std_growth_rate = p['std_growth_rate']
    carrying_capacity = p['carrying_capacity']
    mean_tumor_diameter = p['mean_tumor_diameter']
    std_tumor_diameter = p['std_tumor_diameter']

    with open(csv_path, mode='w') as f:

        writer = csv.writer(f, delimiter=',')

        tumor_diameter = pop_manager.sample_lognormal_param(
            mean=mean_tumor_diameter, std=std_tumor_diameter, retval=pop_manager.patient_size, lowerbound=0.3, upperbound=5)

        growth_rate = pop_manager.sample_normal_param(
            mean=mean_growth_rate, std=std_growth_rate, retval=pop_manager.patient_size, lowerbound=0, upperbound=None)


        for num in range(pop_manager.patient_size):

            writer.writerow([tumor_diameter[num], growth_rate[num], carrying_capacity])

    return csv_path


if __name__ == "__main__":
    pop_man = PropertyManager(10)
    size = pop_man.get_patient_size()
    data = pop_man.sample_lognormal_param(2.5, 2.5, 10000, 0.3, 5)
    plt.hist(data)
    plt.show()
