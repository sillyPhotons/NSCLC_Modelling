"""
Author: Ruiheng Su 2020

Script for reproducing results in Geng's paper:
    Geng, C., Paganetti, H. & Grassberger, C. Prediction of Treatment Response
    for Combined Chemo- and Radiation Therapy for Non-Small Cell Lung Cancer
    Patients Using a Bio-Mathematical Model. Sci Rep 7, 13542 (2017).
    https://doi.org/10.1038/s41598-017-13646-z

Reproduce Treatment Response for > 1 Patients
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

    sampling_range = [0, 60]
    monte_carlo_patient_size = 10
    pop_manager = gp.PropertyManager(monte_carlo_patient_size)
    res_manager = ResultManager()

    # for stage in c.REFER_TUMOR_SIZE_DIST.keys():
    for stage in [1]:
        params = pop_manager.get_param_object_for_radiation()
        x, tumor_volume = \
            pp.Radiation_Treatment_Response_Multiple(params,
                                                     np.arange(
                                                         sampling_range[0]*31, sampling_range[1]*31 + c.RESOLUTION, c.RESOLUTION),
                                                     pop_manager,
                                                     m.rk4_tumor_volume)

        for i in range(monte_carlo_patient_size):
            res_manager.record_prediction(
                ResultObj(plt.plot, x*31., pop_manager.get_diameter_from_volume(tumor_volume[i]), xdes="Days",
                          ydes="Tumor Diameter [$cm$]", curve_label="Radiotherapy Only", label="Radiotherapy Only", color="black", alpha=0.7,),
                comment="Radiotherapy[{}]".format(i)
            )
