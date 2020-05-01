"""
Author: Ruiheng Su 2020

Script for reproducing results in Geng's paper:
    Geng, C., Paganetti, H. & Grassberger, C. Prediction of Treatment Response
    for Combined Chemo- and Radiation Therapy for Non-Small Cell Lung Cancer
    Patients Using a Bio-Mathematical Model. Sci Rep 7, 13542 (2017).
    https://doi.org/10.1038/s41598-017-13646-z

TREATMENT RESPONSE FOR 1 Patient
"""

if __name__ == '__main__':

    import ray
    import time
    import logging
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Minimizer, Parameters, report_fit

    # User defined modules
    import Model as m
    import Constants as c
    import ReadData as rd
    import ParallelPredict as pp
    import GetProperties as gp
    from Result import ResultObj, ResultManager

    plt.rc("text", usetex=True)
    plt.rcParams.update({'font.size': 18,
                         'figure.autolayout': True })
    # configure logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M', level=logging.INFO)
    # initialize ray module for concurrency
    ray.init()

    sampling_range = [0, 7]
    monte_carlo_patient_size = 1
    pop_manager = gp.PropertyManager(monte_carlo_patient_size)
    res_manager = ResultManager()
    
    #####
    V0 = pop_manager.get_volume_from_diameter(8)
    rho = 0.008
    K = pop_manager.get_volume_from_diameter(30)
    alpha = 0.3
    beta = 0.03
    delay_days = [14]
    x = np.arange(
        sampling_range[0]*31, sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION)
    func = m.tumor_volume_GENG
    #####
    x1, tumor_volume1 = pp.Radiation_Response(V0,
                                            rho,
                                            K,
                                            alpha,
                                            beta,
                                            delay_days,
                                            x,
                                            pop_manager,
                                            func)

    plt.plot(x1*31., pop_manager.get_tumor_cell_number_from_volume(tumor_volume1),
             label="Geng", color="black", alpha=0.7, linewidth=3)

    # c.TOTAL_DOSE = 60
    # c.RAD_DOSE = 2
    # c.SCHEME = [3, 4]
    func = m.euler_tumor_volume
    c.RESOLUTION = 0.5
    x = np.arange(
        sampling_range[0]*31, sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION)
    x2, tumor_volume2 = pp.Radiation_Response(V0,
                                            rho,
                                            K,
                                            alpha,
                                            beta,
                                            delay_days,
                                            x,
                                            pop_manager,
                                            func)

    plt.plot(x2*31., pop_manager.get_tumor_cell_number_from_volume(tumor_volume2),
             label="Forward Euler", alpha=0.7, linewidth=3)

    func = m.euler_tumor_volume
    c.RESOLUTION = 0.1
    x = np.arange(
        sampling_range[0]*31, sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION)
    x3, tumor_volume3 = pp.Radiation_Response(V0,
                                            rho,
                                            K,
                                            alpha,
                                            beta,
                                            delay_days,
                                            x,
                                            pop_manager,
                                            func)

    # c.TOTAL_DOSE = 120
    # c.RAD_DOSE = 2
    # c.SCHEME = [4, 2]
    # x3, tumor_volume3 = pp.Radiation_Response(V0,
    #                                         rho,
    #                                         K,
    #                                         alpha,
    #                                         beta,
    #                                         delay_days,
    #                                         x,
    #                                         pop_manager,
    #                                         func)

   

   
    plt.plot(x3*31., pop_manager.get_tumor_cell_number_from_volume(tumor_volume3),
             label="Runge-Kutta", alpha=0.7, linewidth=3)
    plt.title("$\Delta t = {}$".format(c.RESOLUTION))
    plt.xlabel("Time [days]")
    plt.ylabel("Number of Tumor Cells")
    # plt.yscale("log")
    plt.legend()
    plt.show()
    # plt.savefig(res_manager.directory_path + "/response.pdf")

    # res_manager.record_prediction(
    #     ResultObj(plt.plot, x*31., pop_manager.get_diameter_from_volume(tumor_volume), xdes="Days",
    #               ydes="Tumor Diameter [$cm$]", curve_label="Radiotherapy Only", label="Radiotherapy Only", color="black", alpha=0.7,),
    #     comment="Radiotherapy"
    # )
