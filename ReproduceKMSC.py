"""
Author: Ruiheng Su 2020

Script for reproducing results in Geng's paper: 
    Geng, C., Paganetti, H. & Grassberger, C. Prediction of Treatment Response 
    for Combined Chemo- and Radiation Therapy for Non-Small Cell Lung Cancer 
    Patients Using a Bio-Mathematical Model. Sci Rep 7, 13542 (2017). 
    https://doi.org/10.1038/s41598-017-13646-z
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

    sampling_range = [0, 59]
    monte_carlo_patient_size = 10000
    pop_manager = gp.PropertyManager(monte_carlo_patient_size)
    res_manager = ResultManager()

    # for stage in c.REFER_TUMOR_SIZE_DIST.keys():
    for stage in [1]:
        params = pop_manager.get_param_object_for_radiation()

        x, data = rd.read_file("./Data/radiotherapy.csv", interval=sampling_range)
        # x, data = rd.read_file("./Data/stage{}Better.csv".format(stage), interval=sampling_range)

        # vdts = pp.predict_VDT(params, np.arange(
        #     sampling_range[0], sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION), pop_manager, m.tumor_volume_GENG)

        # plt.hist(vdts, 50, range = [0, 500],density=True, alpha=0.7, rwidth=0.85, align='left')
        # plt.show()

        px, py = pp.KMSC_With_Radiotherapy(params,
                                        np.arange(
                                            sampling_range[0]*31, sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION),
                                        pop_manager,
                                        m.tumor_volume_GENG)

        res_manager.record_prediction(
            ResultObj(plt.scatter, x, data, "Months",
                    "Proportion of Patients Alive", curve_label="Radiotherapy Data", label="Radiotherapy Data", color="black", alpha=0.7, s=4),
            ResultObj(plt.step, px, py, "Months",
                    "Proportion of Patients Alive",
                    curve_label="Radiotherapy Model",
                    label="Radiotherapy Model", alpha=0.7),
            # ResultObj(plt.step, x, data, "Months",
            #           "Proportion of Patients Alive", curve_label="Stage {} Data".format(stage), label="Stage {} Data".format(stage), color="black", alpha=0.7),
            # ResultObj(plt.step, px, py, "Months",
            #           "Proportion of Patients Alive",
            #           curve_label="Stage {} Model".format(stage),
            #           label="Stage {} Model".format(stage), alpha=0.7),
            # comment="Stage_[{}]".format(stage)
            comment="Radiotherapy"
        )
