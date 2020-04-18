import unittest

# Modules to be tested
import Model
import Result
import ReadData
import CostFunction
import GetProperties
import ParallelReproduce


class Test_Model (unittest.TestCase):

    def test_discrete_time_models(self):

        retval = Model.discrete_time_tumor_volume_GENG(1, 1, 1, 0, 0, 0, 0)
        self.assertEqual(retval, 1)
        retval = Model.rk4_tumor_volume(1, 1, 1, 0, 0, 0, 0)
        self.assertEqual(retval, 1)
        ret_val = Model.euler_tumor_volume(1, 1, 1, 0, 0, 0, 0)
        self.assertEqual(ret_val, 1)

        self.assertRaises(AssertionError,
                          Model.discrete_time_tumor_volume_GENG, 0, 1, 1, 0, 0, 0, 0)
        self.assertRaises(AssertionError,
                          Model.rk4_tumor_volume, 0, 1, 1, 0, 0, 0, 0)
        self.assertRaises(AssertionError,
                          Model.euler_tumor_volume, 0, 1, 1, 0, 0, 0, 0)

    def test_volume_doubling_time(self):

        self.assertRaises(AssertionError, Model.volume_doubling_time, 1, 1)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 0, 1)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 1, 0)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 0, 0)


class Test_Result (unittest.TestCase):

    def test_ResultObj(self):
        self.assertRaises(AssertionError, Result.ResultObj, None, [0], [0, 0])


class Test_Read_Data (unittest.TestCase):

    def test_exponential(self):

        import numpy as np

        retval = ReadData.exponential(0, 0)
        self.assertEqual(retval, 1)
        retval = ReadData.exponential(np.array([0, 0]), 1)
        self.assertEqual(retval[0], 1)
        self.assertEqual(retval[1], 1)

    def test_read_file(self):

        import numpy as np

        x, y = ReadData.read_file("./Data/radiotherapy.csv", interval=[0, 0])
        self.assertEqual(x[0], 0.)
        self.assertEqual(y[0], 1.)
        self.assertEqual(x.size, 1)
        self.assertEqual(y.size, 1)

        x, y = ReadData.read_file("./Data/radiotherapy.csv", interval=[20, 30])
        self.assertEqual(y.size, x.size)
        self.assertAlmostEqual(x[0], 20, delta=0.5)
        self.assertAlmostEqual(x[-1], 30, delta=0.5)

        x, y = ReadData.read_file(
            "./Data/radiotherapy.csv", interval=np.array([20, 30]))
        self.assertEqual(y.size, x.size)
        self.assertAlmostEqual(x[0], 20, delta=0.5)
        self.assertAlmostEqual(x[-1], 30, delta=0.5)

        x, y = ReadData.read_file(
            "./Data/radiotherapy.csv", interval=np.array([100, 150]))
        self.assertEqual(y.size, x.size)
        self.assertEqual(y.size, 0)

        x, y = ReadData.read_file(
            "./Data/radiotherapy.csv", interval=np.array([0, 150]))
        self.assertEqual(y.size, x.size)
        self.assertEqual(y.size, 60)

        x, y = ReadData.read_file(
            "./Data/radiotherapy.csv", interval=np.array([0, 30]))
        self.assertEqual(y.size, x.size)
        self.assertAlmostEqual(x[0], 0, delta=0.5)
        self.assertAlmostEqual(x[-1], 30, delta=0.5)

    def test_get_fitted_data(self):

        import numpy as np

        self.assertRaises(AssertionError, ReadData.get_fitted_data,
                          np.array([]), np.array([]), 1)
        self.assertRaises(AssertionError, ReadData.get_fitted_data,
                          np.array([1, 2, 3]), np.array([1, 2, 3]), 0)


class Test_GetProperties (unittest.TestCase):

    def setUp(self):
        self.pop_man = GetProperties.PropertyManager(10)

    def test_init(self):
        self.assertRaises(AssertionError, GetProperties.PropertyManager, 0.44)

    def test_sample_normal_param(self):

        import numpy as np
        
        self.assertRaises(AssertionError, self.pop_man.sample_normal_param,\
            0, 1, retval = 1, upperbound = 0,  lowerbound = 0)
        self.assertRaises(AssertionError, self.pop_man.sample_normal_param,\
            0, 1, retval = 1, upperbound = 0,  lowerbound = 10)
        
        retval = self.pop_man.sample_normal_param(0,0,0)
        self.assertEqual(len(retval), 0)
        retval = self.pop_man.sample_normal_param(0,0,0,0,10)
        self.assertEqual(len(retval), 0)
        retval = self.pop_man.sample_normal_param(1,0,10,0,10)
        self.assertEqual(len(retval), 10)
    
    def test_sample_lognormal_param(self):

        import numpy as np
        
        self.assertRaises(AssertionError, self.pop_man.sample_lognormal_param,\
            0, 1, retval = 1, upperbound = 0,  lowerbound = 0)
        self.assertRaises(AssertionError, self.pop_man.sample_lognormal_param,\
            0, 1, retval = 1, upperbound = 0,  lowerbound = 10)
        
        retval = self.pop_man.sample_lognormal_param(0,0,0)
        self.assertEqual(len(retval), 0)
        retval = self.pop_man.sample_lognormal_param(0,0,0,0,10)
        self.assertEqual(len(retval), 0)
        retval = self.pop_man.sample_lognormal_param(1,0,10,0,10)
        self.assertEqual(len(retval), 10)
    
    def test_sample_correlated_params(self):
        param1 = [0,0,0,0]
        param2 = [0,0,0,0]
        corr = 0
        samples = self.pop_man.sample_correlated_params(\
            param1, param2, corr, retval=0)
        self.assertTrue(samples.size == 0)

        param1 = [0,0,None,0]
        param2 = [0,0,None,0]
        corr = 0
        self.assertRaises(AssertionError, self.pop_man.sample_correlated_params, param1, param2, corr)

    def test_get_patient_size (self):

        self.assertEqual(self.pop_man.patient_size, self.pop_man.get_patient_size())

    def test_get_volume_from_diameter (self):
        
        import numpy as np

        retval = self.pop_man.get_volume_from_diameter(np.array([2]))
        expected = (4./3.)*np.pi
        self.assertEqual(retval, expected)


if __name__ == '__main__':
    unittest.main(verbosity=1)
