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

import requests
import time
from helpers import *


# Erstelle eine HTTP-Session
session = requests.Session()

# Erstelle eine Liste zur Speicherung der DMX-Werte
dmx_values = ['0'] * 512


def set_dmx_value(address, value):
    # Aktualisiere den entsprechenden Wert in der Liste
    dmx_values[address - 1] = str(value)

    # Sende die gesamte DMX-Liste als einen String
    session.post('http://localhost:9090/set_dmx', data={'u': '0', 'd': ','.join(dmx_values)})
