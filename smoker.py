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

# Required Libraries
import RPi.GPIO as GPIO
import time
from time import sleep
from rpi_rf import RFDevice
import threading
from threading import Lock


# Global variables
gpio_pin = 23  # GPIO pin connected to the FS1000A transmitter's data pin
code_on = 4543756  # Decimal code to turn the smoke machine on
code_off = 4543792  # Decimal code to turn the smoke machine off
pulse_length = 370  # Pulse length in microseconds
rfdevice = RFDevice(gpio_pin)  # Initialize the RF device for communication
smoke_status = "off"  # Tracks the current state of the smoke machine, default is 'off'

smoke_lock = Lock()


def run_in_thread(fn):
    """Dieser Dekorator führt die gegebene Funktion in einem neuen Thread aus."""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread  # Optional, falls du den Thread später verwalten möchtest

    return wrapper


def init_smoke():
    """Initializes the RF device for smoke machine control."""
    rfdevice.enable_tx()  # Enables the transmission mode


@run_in_thread
def smoke_on():
    global smoke_status
    if smoke_status != "on":
        rfdevice.tx_code(code_on, 1, pulse_length, 24)
        sleep(0.37)
        rfdevice.tx_code(code_on, 1, pulse_length, 24)
        sleep(0.37)
        print("smoke on")
        smoke_status = "on"
        with smoke_lock:
            smoke_status = "on"


@run_in_thread
def smoke_off():
    global smoke_status
    if smoke_status != "off":
        rfdevice.tx_code(code_off, 1, pulse_length, 24)
        sleep(0.37)
        rfdevice.tx_code(code_off, 1, pulse_length, 24)
        sleep(0.37)
        print("smoke off")
        smoke_status = "off"
        with smoke_lock:
            smoke_status = "off"


def cleanup_smoke():
    """Cleans up the RF device after usage."""
    rfdevice.cleanup()  # Resets the RF device to a neutral state
