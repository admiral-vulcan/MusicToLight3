import time
from settings import *
from dmx import *

# Definiere Konstanten
FIRST_X_HOME = 0
FIRST_Y_HOME = 0
SECOND_X_HOME = 0
SECOND_Y_HOME = 0


def scan_closed(num, sec=None):
    address = calc_address(num)
    set_dmx_value(address + 5, 0)
    if sec is not None:
        time.sleep(sec)
        set_dmx_value(address + 5, 20)


def scan_opened(num):
    address = calc_address(num)
    set_dmx_value(address + 5, 20)


def scan_strobe(num, sec=None):
    address = calc_address(num)
    set_dmx_value(address + 5, 250)
    if sec is not None:
        time.sleep(sec)
        set_dmx_value(address + 5, 20)


def scan_axis(num, x, y):
    address = calc_address(num)
    if x < 5:
        x = 5  # Limit to avoid sound
    set_dmx_value(address + 1, x)
    set_dmx_value(address + 2, y)


def scan_gobo(num, go, rotation):
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
    x_home = FIRST_X_HOME if num == 1 else SECOND_X_HOME
    y_home = FIRST_Y_HOME if num == 1 else SECOND_Y_HOME

    address = calc_address(num)
    set_dmx_value(address + 1, 255)
    set_dmx_value(address + 2, 255)
    set_dmx_value(address + 3, 255)
    set_dmx_value(address + 4, 255)
    set_dmx_value(address + 5, 255)
    set_dmx_value(address + 6, 255)
    time.sleep(1.5)

    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)
    set_dmx_value(address + 3, 30)
    set_dmx_value(address + 4, 30)
    set_dmx_value(address + 5, 29)
    set_dmx_value(address + 6, 4)
    time.sleep(1.5)


def scan_go_home(num):
    x_home = FIRST_X_HOME if num == 1 else SECOND_X_HOME
    y_home = FIRST_Y_HOME if num == 1 else SECOND_Y_HOME

    address = calc_address(num)

    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)


last_color = {}


def scan_color(num, scan_color):
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
        print(last_color[num], target_color)
        for color_index in range(current_color_index, target_color_index + step, step):
            current_color = color_values_list[color_index]

            # Send the DMX command to change the color
            set_dmx_value(address + 3, current_color)

            # Remember the current color
            last_color[num] = current_color

            # Wait for 100ms before the next color change
            time.sleep(0.5)


def scan_color_by_value(num, scan_color):
    address = calc_address(num)

    if scan_color < 0:
        scan_color = scan_color * -1
    while scan_color > 255:
        scan_color = scan_color / 10

    set_dmx_value(address + 3, scan_color)
