import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from Constants import TUMOR_DENSITY
import csv


class PropertyManager ():
    """
    PropertyManager class, each PropertyManager object is associated with a 
    single property, patient_size, which is an integer.

    It is passed to the cost function and other models provide sampling 
    methods from normal and lognomral distributions, and other conversion 
    methods to convert between values of tumor volume, diameter and cell number
    """

    def __init__(self, patient_size):
        self.patient_size = patient_size
        self.count = 0  # utility variable

    def sample_normal_param(self, mean, std, retval=1, upperbound=None, lowerbound=None):
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

    def sample_lognormal_param(self, mean, std, retval=1, lowerbound=None, upperbound=None):
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

    def sample_correlated_params(self, param1, param2, corr, retval=1):
        """
        Given (mean, sigma, lowerbound, upperbound) of two values, and linear correlation coefficient, samples from a multivariate normal distribution a number of ordered pairs equal to `retval`. The sampled pairs are returned in a single numpy array with dimensions (2 by retval)

        `param1`: numpy array with 2 elements. `param1[0]` is the mean of the parameter, and `param1[1]` is the standard deviation
        `param2`: numpy array with 2 elements. `param2[0]` is the mean of the parameter, and `param2[1]` is the standard deviation
        `corr`: linear correlation coefficient of `param1` and `param2`. corr = (covariance of param1 and param2)/((sigma of param1)*(sigma of param 2))
        `retval`: number of ordered pairs to return 
        """
        sigma1 = param1[1]
        sigma2 = param2[1]
        lb1, ub1 = param1[2, 3]
        lb2, ub2 = param1[2, 3]
        print(lb1, ub1)
        covariance_matrix = np.array([sigma1**2, corr * sigma1 * sigma2],
                                     [corr * sigma1 * sigma2, sigma2**2])

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

        return 4. / 3. * np.pi * (diameter_array / 2.) ** 3

    def get_diameter_from_volume(self, volume_array):

        return np.cbrt((volume_array / (4./3. * np.pi))) * 2.

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

            writer.writerow(
                [tumor_diameter[num], growth_rate[num], carrying_capacity])

    return csv_path


if __name__ == "__main__":

    from Constants import REFER_TUMOR_SIZE_DIST, TABLE2
    from scipy.stats import truncnorm
    plt.rc("text", usetex=True)
    plt.rcParams['font.family'] = 'serif'

    pop_man = PropertyManager(100000)
    size = pop_man.get_patient_size()

    for stage in REFER_TUMOR_SIZE_DIST.keys():

        mu = np.log(TABLE2[stage][1])
        sigma = np.sqrt(2*(np.abs(np.log(TABLE2[stage][0]) - mu)))
        lb = REFER_TUMOR_SIZE_DIST[stage][2]
        ub = REFER_TUMOR_SIZE_DIST[stage][3]

        print(mu, sigma)

        # lowerbound = (np.log(lb) - mu) / sigma
        # upperbound = (np.log(up) - mu) / sigma

        # norm_rvs = truncnorm.rvs(lowerbound, upperbound, size=size)
        # initial_diameter = list(np.exp((norm_rvs * sigma) + mu))
        # plt.hist(initial_diameter, int(np.ceil(ub - lb)),
        #          density=True, range=(0, up), alpha=0.7, rwidth=0.95)
        # plt.title("Stage {} Volume Distribution".format(stage))
        # plt.xlabel("Tumor Diameter [cm]")
        # plt.ylabel("Frequency")
        # plt.savefig("stage{}.pdf".format(stage))
        # plt.close()

        initial_diameter = pop_man.sample_lognormal_param(mu,
                                                          sigma,
                                                          retval=size,
                                                          lowerbound=lb,
                                                          upperbound=ub)

        plt.hist(initial_diameter, int(np.ceil(ub - lb)),
                 density=True, range=(0, ub), alpha=0.7, rwidth=0.95)
        plt.title("Stage {} Volume Distribution".format(stage))
        plt.xlabel("Tumor Diameter [cm]")
        plt.ylabel("Frequency")
        plt.savefig("stage{}.pdf".format(stage))
        plt.close()