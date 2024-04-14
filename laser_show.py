# MusicToLight3  Copyright (C) 2024  Felix Rau.
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

from helpers import *
from dmx import *
import time


def set_laser_show(adr, num):
    # Definiere die Adressen für die verschiedenen Parameter
    start_adr = calc_address(6)

    # Setze die Werte für die verschiedenen Parameter
    set_dmx_value(start_adr + adr, num)


def laser_off():
    set_laser_show(0, 0)
    set_laser_show(3, 0)


def laser_bird_start():
    set_laser_show(3, 129)
    set_laser_show(4, 126)  # 127
    time.sleep(0.1)
    set_laser_show(0, 0)  # reset first!
    size = 125
    for n in range(0, 14):
        for x in range(0, 9):
            print(size)
            set_laser_show(4, size)  # 127
            set_laser_show(3, x + 129)
            set_laser_show(0, 1)
            time.sleep(0.1)
            size -= 1


def laser_bird1():
    set_laser_show(3, 129)
    time.sleep(0.1)
    set_laser_show(0, 0)  # reset first!
    for x in range(130, 138):
        set_laser_show(3, x)
        set_laser_show(0, 1)
        time.sleep(0.1)


def laser_bird2():
    set_laser_show(3, 151)
    time.sleep(0.1)
    set_laser_show(0, 0)  # reset first!
    for x in range(152, 160):
        set_laser_show(3, x)
        set_laser_show(0, 1)
        time.sleep(0.1)


def laser_bird_between():
    set_laser_show(0, 1)
    for x in range(139, 160):
        set_laser_show(3, x)
        time.sleep(0.1)


def laser_bird_end():
    set_laser_show(0, 1)
    for x in range(161, 182):
        set_laser_show(3, x)
        time.sleep(0.1)


def laser_slow_dance():
    start_adr = calc_address(6)
    values = [
        1, 0, 255, 56, 50, 33, 0, 139, 0, 160,
        0, 198, 0, 255, 0
    ]
    set_dmx_values_from(start_adr, values)


def laser_fast_dance(x, y, color):
    c = laser_color_to_int(color)
    x = x/4 + 1
    y = y/4 + 1
    start_adr = calc_address(6)
    values = [
        1, 0, 255,
        56, x, y,
        y, 0, x, 0,
        0,
        c,
        0, 0, 0
    ]
    set_dmx_values_from(start_adr, values)


def laser_star_chase():
    start_adr = calc_address(6)
    values = [
        1, 0, 255,
        78, 0, 128, 178, 0, 0, 0,
        0,
        36,  # white-ish
        0, 0, 0
    ]
    set_dmx_values_from(start_adr, values)


# laser_slow_dance()

"""
laser_bird_start()

laser_bird_between()

laser_bird2()
laser_bird2()
laser_bird2()
laser_bird2()

laser_bird_end()

laser_off()
"""
