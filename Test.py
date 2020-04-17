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

        retval = Model.discrete_time_tumor_volume_GENG(1,1,1,0,0,0,0)
        self.assertEqual(retval, 1)
        retval = Model.rk4_tumor_volume(1,1,1,0,0,0,0)
        self.assertEqual(retval, 1)
        ret_val = Model.euler_tumor_volume(1,1,1,0,0,0,0)
        self.assertEqual(ret_val, 1)

        self.assertRaises(AssertionError, \
            Model.discrete_time_tumor_volume_GENG, 0,1,1,0,0,0,0)
        self.assertRaises(AssertionError, \
            Model.rk4_tumor_volume, 0,1,1,0,0,0,0)
        self.assertRaises(AssertionError, \
            Model.euler_tumor_volume, 0,1,1,0,0,0,0)

    def test_volume_doubling_time(self):

        self.assertRaises(AssertionError, Model.volume_doubling_time, 1,1)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 0,1)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 1,0)
        self.assertRaises(AssertionError, Model.volume_doubling_time, 0,0)

class Test_Result (unittest.TestCase):
    
    def setUp(self):
        self.res_man = Result.ResultManager()
    
    def test_ResultObj(self):
        self.assertRaises(AssertionError, Result.ResultObj, None, [0], [0,0])

if __name__ == '__main__':
    unittest.main()