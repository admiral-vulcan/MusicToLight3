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

# Global variables
gpio_pin = 23  # GPIO pin connected to the FS1000A transmitter's data pin
code_on = 4543756  # Decimal code to turn the smoke machine on
code_off = 4543792  # Decimal code to turn the smoke machine off
pulse_length = 370  # Pulse length in microseconds
rfdevice = RFDevice(gpio_pin)  # Initialize the RF device for communication
smoke_status = "off"  # Tracks the current state of the smoke machine, default is 'off'


def init_smoke():
    """Initializes the RF device for smoke machine control."""
    rfdevice.enable_tx()  # Enables the transmission mode


def smoke_on():
    """Turns the smoke machine on by sending the 'on' code."""
    global smoke_status  # Access the global variable
    if smoke_status != "on":  # Check if the machine is not already on
        rfdevice.tx_code(code_on, 1, pulse_length, 24)
        rfdevice.tx_code(code_on, 1, pulse_length, 24)
        rfdevice.tx_code(code_on, 1, pulse_length, 24)
        # print("Smoke machine turned on")
        smoke_status = "on"  # Update the status
        sleep(0.37)


def smoke_off():
    """Turns the smoke machine off by sending the 'off' code."""
    global smoke_status  # Access the global variable
    if smoke_status != "off":  # Check if the machine is not already off
        rfdevice.tx_code(code_off, 1, pulse_length, 24)
        rfdevice.tx_code(code_off, 1, pulse_length, 24)
        rfdevice.tx_code(code_off, 1, pulse_length, 24)
        # print("Smoke machine turned off")
        smoke_status = "off"  # Update the status
        sleep(0.37)


def cleanup_smoke():
    """Cleans up the RF device after usage."""
    rfdevice.cleanup()  # Resets the RF device to a neutral state
