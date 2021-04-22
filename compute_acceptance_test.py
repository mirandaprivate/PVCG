# pylint: disable=missing-docstring
import unittest
import pathlib

from compute_acceptance import compute_acceptance

THIS_DIR = pathlib.Path(__file__).parent

def read_data():
    '''Read the rondomly generated data.

    The test_data is the file and the data in it is I randomly generated.
    contains 50 sets of data,each set has five rows, the first line is '(',
    the second line is effective contribution data,the third line is empty,
    the fourth line is cost type data the fifth line is ')'.

    Returns:
        This function return two sets of numbers, each set consisting of
        50 sefs of randomly generated arrays.
    '''
    with open(THIS_DIR / 'test_data', 'r') as f:
        lines = f.read().splitlines()
        data = []
        for num, line in enumerate(lines):
            if line == '(':
                data.append(lines[num + 1:num + 4])
        accept, cost = [], []
        # accept the data with two lists. Index[0] is the data for effective
        # contribution, index[2] is the data for cost_type.
        for index in data:
            accept.append(index[0])
            cost.append(index[2])
        return change_str_to_float(accept), change_str_to_float(cost)

def change_str_to_float(string):
    ''' change the string to the float which in the list accept or list cost.
    '''

    test_data = []
    for i in string:
        index = [float(l) for l in i.split(',')]
        test_data.append(index)
    return test_data

class TestComputeAcceptance(unittest.TestCase):

    def test_compute_acceptance_max(self):

        data1, data2 = read_data()
        for i in range(50):
            alpha = 25
            actual = compute_acceptance(data1[i], data2[i], alpha=alpha)
            print(actual)
            expected = ([1.0] * 10)
            self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
