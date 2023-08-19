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


# Initialize GPIO settings for smoke machine control
def init_smoke():
    """Initialize GPIO pin for smoke machine control."""
    GPIO.setmode(GPIO.BCM)  # Set GPIO numbering mode to BCM
    GPIO.setup(23, GPIO.OUT)  # Set GPIO pin 23 as an output


# Activate smoke machine
def smoke_on():
    """Turn the smoke machine on."""
    GPIO.output(23, GPIO.HIGH)  # Set GPIO pin 23 to high, activating the relay


# Deactivate smoke machine
def smoke_off():
    """Turn the smoke machine off."""
    GPIO.output(23, GPIO.LOW)  # Set GPIO pin 23 to low, deactivating the relay


# Cleanup GPIO settings
def cleanup_smoke():
    """Cleanup GPIO settings after usage."""
    GPIO.cleanup()  # Reset GPIO settings to default
