import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from Constants import STAGE, TUMOR_DENSITY

"""
"""
class PropertyManager ():

    patient_size = None

    def __init__(self, patient_size):
        PropertyManager.patient_size = patient_size
        return

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
            
    def get_patient_size (self):

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

        volume_array = self.get_volume_from_tumor_cell_number(cell_number_array)
        diameter_array = self.get_diameter_from_volume(volume_array)

        return diameter_array

if __name__ == "__main__":
    pop_man = PropertyManager(10)
    size = pop_man.get_patient_size()
    data = pop_man.sample_lognormal_param(2.5, 2.5, 10000, 0.3, None)
    plt.hist(data)
    plt.show()