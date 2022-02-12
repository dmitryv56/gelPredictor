#!/usr/bin/env python3

""" There is a simple example of  testing by using 'unitest'-framework.
This test checks what field size are greater then garden size.
In oreder to run this test in command line:
 cd <path to project>/RosenGarden
 python3 -m unittest testing/simple_test.py

"""

import unittest
import os
from unittest import TestCase
from main import init_task

from src.rosesfield import RosesField

class TestGarden(TestCase):
    def test_init_good(self):
        field_width, field_height, garden_width, garden_height, purchased_squares, location_roses, location_costs = \
            init_task()

        roses_field = RosesField(field_width, field_height, purchased_squares, location_roses, location_costs,
                                 garden_width,
                                 garden_height)

        self.assertEqual(roses_field.field_width,field_width)
        self.assertEqual(roses_field.field_height, field_height)
        self.assertEqual(roses_field.garden_width, garden_width)
        self.assertEqual(roses_field.garden_height, garden_height)
        self.assertGreaterEqual(roses_field.field_width, roses_field.garden_width)
        self.assertGreaterEqual(roses_field.field_height, roses_field.garden_height)
        # dedicated assertion
        # self.assertGreaterEqual(roses_field.garden_height,roses_field.field_height)

if __name__ == '__main__':
    unittest.main()