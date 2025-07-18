# Project RoboSplatter
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import unittest
import numpy as np
from robo_splatter.utils.direct_transform import direction_transform, convert_direction_string_to_rotation_matrix, parse_direction_string

class TestDirectionTransform(unittest.TestCase):

    def test_parse_direction_string(self):
        self.assertEqual(parse_direction_string("front:+x up:+y"), ("+x", "+y"))
        self.assertEqual(parse_direction_string("front:-z up:+x"), ("-z", "+x"))
        with self.assertRaises(ValueError):
            parse_direction_string("invalid string")

    def test_convert_direction_string_to_rotation_matrix(self):
        expected_matrix = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(
            convert_direction_string_to_rotation_matrix("front:+x up:+z"),
            expected_matrix
        )

        expected_matrix_2 = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(
            convert_direction_string_to_rotation_matrix("front:+z up:+y"),
            expected_matrix_2
        )

        expected_matrix_combined = (expected_matrix@expected_matrix_2)
        np.testing.assert_array_almost_equal(
            convert_direction_string_to_rotation_matrix("front:+x up:+z", "front:+z up:+y"),
            expected_matrix_combined
        )

    def test_direction_transform(self):
        expected_transform = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(
            direction_transform("front:+x up:+y", "front:+x up:+y"),
            expected_transform
        )

        expected_transform_2 = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        np.testing.assert_array_almost_equal(
            direction_transform("front:+x up:+y", "front:+z up:+y"),
            expected_transform_2
        )

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    unittest.main()