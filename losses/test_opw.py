from unittest import TestCase
from losses import OPWMetric
import numpy as np


class TestOPWMetric(TestCase):

    def test_calculate_1d_distances(self):
        loss = OPWMetric()
        a = np.array([[1, 2, 3, 4, 5]]).T
        b = np.array([[1, 2, 3, 4, 5]]).T
        dist, transport = loss(a, b)
        print('Distance : ', dist)
        print('transport: ', transport)
        print('a = trasport.T*b: ', a, '= ', np.dot(transport.T, b))
        print('b = trasport*a: ', b, '= ', np.dot(transport, a))

    def test_calculate_assigment(self):
        loss = OPWMetric()
        a = np.array([[2, 1, 2, 1, 2]]).T
        b = np.array([[1, 3, 5]]).T
        a_assignement_on_b, b_assignement_on_a = loss.calculate_assigment(a,b)
        print(a_assignement_on_b.T, a.T)
        print(b_assignement_on_a.T, b.T)