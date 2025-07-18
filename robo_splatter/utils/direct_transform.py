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


import re
import numpy as np


def direction_transform(direction_from: str, direction_to: str):  # noqa
    """direction transformation matrix

    Args:
        direction_from (str): front:+-[xyz] up:+-[xyz]
        direction_to (str): same format as direction_from
    """
    rot_from2world = convert_direction_string_to_rotation_matrix(
        direction_from
    )  # noqa
    rot_to2world = convert_direction_string_to_rotation_matrix(
        direction_to
    )  # noqa

    rot = rot_to2world.T @ rot_from2world

    return rot


def convert_direction_string_to_rotation_matrix(direction_string: str):  # noqa
    front_str, up_str = parse_direction_string(direction_string)

    # directions
    # ^ +z
    # |
    # | / +y
    # |/
    # +-------> +x
    directions = {
        "-x": [-1, 0, 0],
        "+x": [1, 0, 0],
        "-y": [0, -1, 0],
        "+y": [0, 1, 0],
        "-z": [0, 0, -1],
        "+z": [0, 0, 1],
    }
    front = directions[front_str]
    up = directions[up_str]

    result_matrix = np.stack([front, np.cross(up, front), up], -1).T

    return result_matrix


def parse_direction_string(direction_string):
    pattern = r"front:([\+\-][xyz]) up:([\+\-][xyz])"
    matches = re.findall(pattern, direction_string)
    if len(matches) != 1:
        raise ValueError(
            f"invalid format for direction string {direction_string}"
        )  # noqa

    return matches[0][0], matches[0][1]
