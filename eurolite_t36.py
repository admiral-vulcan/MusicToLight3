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


def set_eurolite_t36(num, red, green, blue, brightness, strobe):
    # Definiere die Adressen für die verschiedenen Parameter
    red_address = calc_address(num)
    green_address = calc_address(num) + 1
    blue_address = calc_address(num) + 2
    brightness_address = calc_address(num) + 3
    strobe_address = calc_address(num) + 4

    # Setze die Werte für die verschiedenen Parameter
    set_dmx_value(red_address, red)
    set_dmx_value(green_address, green)
    set_dmx_value(blue_address, blue)
    set_dmx_value(brightness_address, brightness)
    set_dmx_value(strobe_address, strobe)


def demo_eurolite_t36():
    # Rot von 0 bis 255
    for red in range(0, 256, 5):
        set_eurolite_t36(5, red, 0, 0, 255, 0)

    # Grün von 0 bis 255
    for green in range(0, 256, 5):
        set_eurolite_t36(5, 255, green, 0, 255, 0)

    # Blau von 0 bis 255
    for blue in range(0, 256, 5):
        set_eurolite_t36(5, 255, 255, blue, 255, 0)

    set_eurolite_t36(5, 255, 255, 255, 255, 252)
    time.sleep(2)
    set_eurolite_t36(5, 255, 255, 255, 255, 0)

    # Alle Farben von 255 bis 0
    for value in reversed(range(0, 256, 5)):
        set_eurolite_t36(5, value, value, value, 255, 0)