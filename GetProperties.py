import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from Constants import STAGE, TUMOR_DENSITY
import csv

"""
    PropertyManager class, each PropertyManager object is associated with a 
    single property, patient_size, which is an integer.

    It is passed to the cost function and other models provide sampling 
    methods from normal and lognomral distributions, and other conversion 
    methods to convert between values of tumor volume, diameter and cell number
"""
class PropertyManager ():

    def __init__(self, patient_size):
        self.patient_size = patient_size

    """
        Given the mean and the standard deviation, this method returns an array 
        of samples from the specified normal distribution, with the number of
        samples equal to retval.

        `mean`: mean of the normal distribution
        `std`: standard deviation of the normal distribution
        `retval`: integer number of values to return
        `upperbound`, `lowerbound`: upper and lowe bound of returned values

        Requires: `upperbound` >= `lowerbound` 
    """
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
            
            assert(upperbound >= lowerbound)
            data = list()
            i = 0
            while i < retval:
                point = np.random.normal(mean, std, 1)

                if (lowerbound < point < upperbound):
                    data.append(point[0])
                    i += 1
            return data
    
    """
        Given the mean and the standard deviation, this method returns an array 
        of samples from the specified lognormal distribution, with the number of
        samples equal to retval.

        `mean`: mean of the normal distribution
        `std`: standard deviation of the normal distribution
        `retval`: integer number of values to return
        `upperbound`, `lowerbound`: upper and lowe bound of returned values

        Requires: `upperbound` >= `lowerbound` 
    """
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

        return self.patient_size

    def get_volume_from_diameter(self, diameter_array):

        volume_array = (4.*np.pi/24.) * (diameter_array**3)

        return volume_array

    def get_diameter_from_volume(self, volume_array):

        diameter_array = np.cbrt((24./(4*np.pi)) * volume_array)

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

"""
    Given the location of a csv file, a patient population is generated via 
    random sampling, and the generated csv file has the form that each row 
    represents a patient, each the columns have values representing tumor 
    diameter in cm, growth rate, and carrying capacity in cm

    csv_path: string specifying the path to save the csv file generate, 
              including the name of the csv file
    params: Parameters object containing the following Parameter objects:
        mean_growth_rate = p['mean_growth_rate']
        std_growth_rate = p['std_growth_rate']
        carrying_capacity = p['carrying_capacity']
        mean_tumor_diameter = p['mean_tumor_diameter']
        std_tumor_diameter = p['std_tumor_diameter']
    pop_manager: PropertyManager object
"""
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
