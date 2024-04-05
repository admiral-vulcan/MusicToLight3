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

import threading
import requests
from helpers import *

# Erstelle eine HTTP-Session
session = requests.Session()

# Erstelle eine Liste zur Speicherung der DMX-Werte
dmx_values = ['0'] * 512


def send_dmx_values():
    session.post('http://musictolight-dmx/:9090/set_dmx', data={'u': '0', 'd': ','.join(dmx_values)})


def set_dmx_value(address, value):
    # Prüfe, ob der Wert sich geändert hat
    if dmx_values[address - 1] != str(value):
        dmx_values[address - 1] = str(value)
        threading.Thread(target=send_dmx_values).start()


def set_dmx_values_from(address, values):
    changed = False
    for i, value in enumerate(values):
        if address + i < len(dmx_values) and dmx_values[address + i - 1] != str(value):
            dmx_values[address + i - 1] = str(value)
            changed = True

    if changed:
        threading.Thread(target=send_dmx_values).start()
