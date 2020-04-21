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

    # configure logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M', level=logging.INFO)
    # initialize ray module for concurrency
    ray.init()

    sampling_range = [0, 13]
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

    x, tumor_volume = pp.Radiation_Response(V0,
                                            rho,
                                            K,
                                            alpha,
                                            beta,
                                            delay_days,
                                            x,
                                            pop_manager,
                                            func)

    plt.plot(x*31., pop_manager.get_tumor_cell_number_from_volume(tumor_volume),
             label="Radiotherapy Only", color="black", alpha=0.7,)
    plt.xlabel("Days")
    plt.ylabel("Number of Tumor Cells")
    plt.yscale("log")
    plt.savefig(res_manager.directory_path + "/response.pdf")

    # res_manager.record_prediction(
    #     ResultObj(plt.plot, x*31., pop_manager.get_diameter_from_volume(tumor_volume), xdes="Days",
    #               ydes="Tumor Diameter [$cm$]", curve_label="Radiotherapy Only", label="Radiotherapy Only", color="black", alpha=0.7,),
    #     comment="Radiotherapy"
    # )
