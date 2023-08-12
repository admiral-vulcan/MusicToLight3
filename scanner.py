# MusicToLight3  Copyright (C) 2023  Felix Rau.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
from helpers import *
from dmx import *

# Define constants for scanner home positions
FIRST_X_HOME = 155
FIRST_Y_HOME = 30
FIRST_X_MIN = 110
FIRST_X_MAX = 155
FIRST_Y_MIN = 0
FIRST_Y_MAX = 90

SECOND_X_HOME = 12
SECOND_Y_HOME = 145
SECOND_X_MIN = 5
SECOND_X_MAX = 250
SECOND_Y_MIN = 5
SECOND_Y_MAX = 250


def scan_closed(num, sec=None):
    """
    Close the scanner shutter.

    Arguments:
    num -- the scanner number
    sec -- if provided, the shutter will reopen after this number of seconds
    """
    address = calc_address(num)
    set_dmx_value(address + 5, 0)
    if sec is not None:
        time.sleep(sec)
        set_dmx_value(address + 5, 20)


def scan_opened(num):
    """
    Open the scanner shutter.

    Arguments:
    num -- the scanner number
    """
    address = calc_address(num)
    set_dmx_value(address + 5, 20)


def scan_strobe(num, sec=None):
    """
    Make the scanner strobe.

    Arguments:
    num -- the scanner number
    sec -- if provided, the scanner will stop strobing after this number of seconds
    """
    address = calc_address(num)
    set_dmx_value(address + 5, 250)
    if sec is not None:
        time.sleep(sec)
        set_dmx_value(address + 5, 20)



def scan_axis(num, x, y):
    """
    Move the scanner to the given position.
    We map the values to stay inside the predefined borders.

    Arguments:
    num -- the scanner number
    x, y -- the desired position
    """

    if x < 0:
        x = 0

    if x > 255:
        x = 255

    if y < 0:
        y = 0

    if y > 255:
        y = 255

    if num == 1:
        x = map_value(x, 0, 255, FIRST_X_MIN, FIRST_X_MAX)
        y = map_value(y, 0, 255, FIRST_Y_MIN, FIRST_Y_MAX)
    elif num == 2:
        x = map_value(x, 0, 255, SECOND_X_MIN, SECOND_X_MAX)
        y = map_value(y, 0, 255, SECOND_Y_MIN, SECOND_Y_MAX)

    address = calc_address(num)
    if x < 5:
        x = 5  # Limit to avoid sound
    set_dmx_value(address + 1, x)
    set_dmx_value(address + 2, y)
    time.sleep(0.01)

def scan_gobo(num, go, rotation):
    """
    Set the scanner's gobo and rotation.

    Arguments:
    num -- the scanner number
    go -- the desired gobo
    rotation -- the desired rotation
    """
    address = calc_address(num)
    # Transformation of rotation value
    if rotation > 0:
        rotation = (rotation / 2.2) + 5
    elif rotation < 0:
        rotation = ((rotation * -1) / 2.2) + 140
    else:
        rotation = 0

    set_dmx_value(address + 6, int(rotation))

    gobo_values = {
        1: 50,
        2: 80,
        3: 100,
        4: 150,
        5: 180,
        6: 200,
        7: 250
    }

    set_dmx_value(address + 4, gobo_values.get(go, 0))


def scan_reset(num):
    """
    Reset the scanner to its home position.

    Arguments:
    num -- the scanner number
    """
    x_home = 0
    y_home = 0
    if num == 1:
        x_home = FIRST_X_HOME
        y_home = FIRST_Y_HOME
    elif num == 2:
        x_home = SECOND_X_HOME
        y_home = SECOND_Y_HOME

    address = calc_address(num)

    set_dmx_value(address + 1, 255)
    set_dmx_value(address + 2, 255)
    time.sleep(0.25)
    scan_color(num, 255)
    set_dmx_value(address + 4, 255)
    time.sleep(0.25)
    set_dmx_value(address + 5, 255)
    set_dmx_value(address + 6, 255)
    time.sleep(0.25)

    set_dmx_value(address + 1, 0)
    set_dmx_value(address + 2, 0)
    time.sleep(0.25)
    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)
    time.sleep(0.25)
    scan_color(num, 0)
    set_dmx_value(address + 4, 30)
    time.sleep(0.25)
    set_dmx_value(address + 5, 29)
    set_dmx_value(address + 6, 4)
    time.sleep(0.25)


def scan_go_home(num):
    """
    Move the scanner to its home position.

    Arguments:
    num -- the scanner number
    """
    x_home = FIRST_X_HOME if num == 1 else SECOND_X_HOME
    y_home = FIRST_Y_HOME if num == 1 else SECOND_Y_HOME

    address = calc_address(num)

    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)


# Global dictionary to store the last color set for each scanner
last_color = {}


def scan_color(num, scan_color):
    """
    Change the color of a specified scanner.

    Arguments:
    num -- the number of the scanner to control (1 or 2 currently)
    scan_color -- the target color to change to, can be a string (name of the color) or an int (color value)

    The function first calculates the address based on the scanner number. Then it checks the type of scan_color.
    If it's a string, it fetches the corresponding color value, or defaults to "white".
    If it's an int, it finds the closest predefined color value.

    The function then checks if there's a recorded last color for this scanner. If not, it assumes that this is the
    first call for this scanner, and records the target color as the last color.

    It then calculates the current and target color indices in the predefined color list, and determines the step
    direction based on their relative positions.

    If this is the first call for this scanner or the last color is different from the target color,
    the function steps through the colors from the current color to the target color,
    sends the DMX command to change the color, records the current color as the last color, and waits for 100ms
    before the next color change.
    """
    # Calculate the address based on the scanner number
    address = calc_address(num)
    color_values_list = [30, 50, 80, 100, 150, 180, 200, 250]
    color_values = {
        "white": 30,
        "red": 50,
        "yellow": 80,
        "purple": 100,
        "green": 150,
        "orange": 180,
        "blue": 200,
        "pink": 250,
    }

    if type(scan_color) is str:
        target_color = color_values.get(scan_color, 30)  # Default to white
    else:
        # Find the closest color in the list
        target_color = min(color_values_list, key=lambda x: abs(x - scan_color))

    # Initialize the last color if this is the first call for this scanner
    if num not in last_color:
        last_color[num] = target_color
        first_color = True
    else:
        first_color = False

    current_color_index = color_values_list.index(last_color[num])
    target_color_index = color_values_list.index(target_color)

    step = 1 if current_color_index < target_color_index else -1

    if first_color or last_color[num] != target_color:
        for color_index in range(current_color_index, target_color_index + step, step):
            current_color = color_values_list[color_index]

            # Send the DMX command to change the color
            set_dmx_value(address + 3, current_color)

            # Remember the current color
            last_color[num] = current_color

            # Wait for 100ms before the next color change
            time.sleep(0.5)
