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
    if x < 5: x = 5  # Limit to avoid sound
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

    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)
    set_dmx_value(address + 3, 30)
    set_dmx_value(address + 4, 30)
    set_dmx_value(address + 5, 29)
    set_dmx_value(address + 6, 4)


def scan_go_home(num):
    x_home = FIRST_X_HOME if num == 1 else SECOND_X_HOME
    y_home = FIRST_Y_HOME if num == 1 else SECOND_Y_HOME

    address = calc_address(num)

    set_dmx_value(address + 1, x_home)
    set_dmx_value(address + 2, y_home)


def scan_color(num, scan_color):
    address = calc_address(num)

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

    set_dmx_value(address + 3, color_values.get(scan_color, 30))  # Default to white


def scan_color_by_value(num, scan_color):
    address = calc_address(num)

    if scan_color < 0:
        scan_color = scan_color * -1
    while scan_color > 255:
        scan_color = scan_color / 10

    set_dmx_value(address + 3, scan_color)
