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
        ret_val = Model.rk4_tumor_volume(1,1,1,0,0,0,0)
        self.assertEqual(ret_val, 1)

    def test_volume_doubling_time(self):

        ret_val = volume_doubling_time


if __name__ == '__main__':
    unittest.main()