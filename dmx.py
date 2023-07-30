import requests
import time
from settings import *


# Erstelle eine HTTP-Session
session = requests.Session()

# Erstelle eine Liste zur Speicherung der DMX-Werte
dmx_values = ['0'] * 512


def set_dmx_value(address, value):
    # Aktualisiere den entsprechenden Wert in der Liste
    dmx_values[address - 1] = str(value)

    # Sende die gesamte DMX-Liste als einen String
    session.post('http://localhost:9090/set_dmx', data={'u': '0', 'd': ','.join(dmx_values)})
