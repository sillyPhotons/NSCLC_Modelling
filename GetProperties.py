import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from Constants import STAGE, TUMOR_DENSITY

"""
"""


def sample_dist(stage, num_points=10000):

    dist_params = STAGE[stage]

    if (stage == 1):

        data = [k for k in np.random.lognormal(
        *dist_params, num_points) if 0.3 < k < 5.]
        
        i = len(data)
        while i < num_points:
            point = np.random.lognormal(*dist_params, 1)

            if (stage == 1):
                if (0.3 < point < 5):
                    data.append(point[0])
                    i += 1
    else:
        data = [k for k in np.random.lognormal(
        *dist_params, num_points) if 0.3 < k]

        i = len(data)

        while i < num_points:
            point = np.random.lognormal(*dist_params, 1)

            if (point > 0.3):
                data.append(point[0])
                i += 1

    return np.array(data)


def get_volume_from_diameter(diameter_array):

    volume_array = (4.*np.pi/3.) * (diameter_array/2.)**2

    return volume_array


def get_diameter_from_volume(volume_array):

    diameter_array = np.sqrt((3./(4*np.pi)) * volume_array) * 2

    return diameter_array


def get_tumor_cell_number_from_diameter(diameter_array):

    volume_array = get_volume_from_diameter(diameter_array)
    cell_number_array = volume_array * TUMOR_DENSITY

    return cell_number_array


def get_tumor_cell_number_from_volume(volume_array):

    cell_number_array = volume_array * TUMOR_DENSITY

    return cell_number_array


def get_volume_from_tumor_cell_number(cell_number_array):

    volume_array = cell_number_array / TUMOR_DENSITY

    return volume_array


def get_diameter_from_tumor_cell_number(cell_number_array):

    volume_array = get_volume_from_tumor_cell_number(cell_number_array)
    diameter_array = get_diameter_from_volume(volume_array)

    return diameter_array


def sample_param(mean, std, retval=1):
    return np.random.normal(mean, std, retval)


if __name__ == "__main__":

    data = sample_dist(1)
    print(data)
