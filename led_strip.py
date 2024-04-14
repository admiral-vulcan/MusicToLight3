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

# import cProfile
from rpi_ws281x import Color
from collections import deque
import random
import pyaudio
import numpy as np
from com_udp import *
from helpers import *
from eurolite_t36 import *
import socket
import time


class UDPNeoPixel:
    def __init__(self, num_leds):
        self.num_leds = num_leds
        self.colors = [(0, 0, 0)] * num_leds  # Initialisiere alle LEDs auf ausgeschaltet (schwarz)
        self.ip_address = '192.168.1.153'
        self.port = 4210

    def numPixels(self):
        """ Gibt die Anzahl der LEDs zurück. """
        return self.num_leds

    def setPixelColor(self, i, color):
        """ Setzt die Farbe einer einzelnen LED.

        Args:
            i (int): Index der LED.
            color (tuple or int): RGB-Tupel oder Ganzzahl, die eine Farbe darstellt.
        """
        if isinstance(color, tuple) and len(color) == 3:
            if 0 <= i < self.num_leds:
                self.colors[i] = color
        elif isinstance(color, int):
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            self.colors[i] = (r, g, b)

    def show(self):
        """ Sendet die Farbinformationen aller LEDs über UDP als binäre Daten. """
        message = bytearray(b'mls_')  # Präfix für die Nachricht
        message.extend(self.num_leds.to_bytes(2, byteorder='little'))  # Anzahl der LEDs als zwei Bytes

        for r, g, b in self.colors:
            # Stelle sicher, dass R, G, B Werte korrekt sind
            message.extend([int(r) % 256, int(g) % 256, int(b) % 256])  # Jede Farbe als ein Byte

        # Sende die Nachricht über UDP
        send_udp_message(self.ip_address, self.port, message)


# Global variable for saving the timestamp of the last strip.show() call
last_show_time = 0

# LED-strip configuration:
LED_COUNT = 270  # Anzahl der LED-Pixel.
# LED_PIN = 18  # GPIO-Pin, der mit dem Datenpin des LED-Streifens verbunden ist.
# LED_FREQ_HZ = 800000  # LED-Signalfrequenz in Hz (normalerweise 800khz)
# LED_DMA = 10  # DMA-Kanal, der für die Signalgenerierung verwendet wird (kann 0-14 sein).
# LED_BRIGHTNESS = 255  # Eingestellte Helligkeit (0 für dunkelste und 255 für hellste)
# LED_INVERT = False  # True zum Invertieren des Signals (bei Verwendung von NPN-Transistor-Level-Shifter)
# LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create NeoPixel object with appropriate configuration.
# strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
strip = UDPNeoPixel(LED_COUNT)
# Intialize the library (must be called once before other functions).
# strip.begin()
global_led_state = [(0, 0, 0) for _ in range(strip.numPixels())]

recent_audio_inputs = deque(maxlen=int(3))  # adjust as needed, originally 20

# Global variable to keep track of current LED colors
current_colors = [(0, 0, 0)] * strip.numPixels()


def led_star_chase(c, wait_ms):
    NUMCHASE = 30
    NUMPIX = int(strip.numPixels() / 2)
    for j in range(NUMCHASE):
        startPo = (NUMPIX / NUMCHASE * j) - random.randint(5, 15)
        if startPo < 0:
            startPo = 0
        stopPo = (NUMPIX / NUMCHASE * (j + 1)) + random.randint(5, 15)
        if stopPo > NUMPIX:
            stopPo = NUMPIX

        for q in range(3):
            i = startPo
            while i < stopPo:
                if 0 <= int(i + q) < NUMPIX:  # Überprüfe, ob das Pixel innerhalb der Grenzen liegt
                    strip.setPixelColor(int(i + q), c)
                i += random.randint(1, 15)
            strip.show()

            time.sleep(wait_ms / 1000.0)

            for i in range(0, NUMPIX, 3):
                if 0 <= int(i + q) < NUMPIX:  # Überprüfe, ob das Pixel innerhalb der Grenzen liegt
                    strip.setPixelColor(int(i + q), 0)

        strip.show()  # Sende die Farbänderungen an den Streifen
        for i in range(int(strip.numPixels() / 2)):
            strip.setPixelColor(i, 0)  # Schalte alle LEDs aus
        strip.show()  # Sende die Farbänderungen an den Streifen
    set_eurolite_t36(5, 0, 0, 0, 255, 0)


def led_set_all_pixels_color(red, green, blue):
    """
    Set the color of all pixels on the strip.

    Args:
    - strip: The LED strip object.
    - red, green, blue: Integers representing the color values.
    """

    # Konvertiere die RGB-Werte in eine 32-Bit-Farbe
    color = (red << 16) | (green << 8) | blue

    # Gehe durch alle Pixel und setze ihre Farbe
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)

    # Aktualisiere den LED-Streifen, um die Änderungen anzuzeigen
    strip.show()


def smooth_transition(current, target, step=5):
    """Smoothly transition from current color to target color."""
    r = int(current[0] + min(max(target[0] - current[0], -step), step))
    g = int(current[1] + min(max(target[1] - current[1], -step), step))
    b = int(current[2] + min(max(target[2] - current[2], -step), step))
    return r, g, b


@led_in_thread
def led_color_flow(pos, audio_input, reduce=2, first_color="blue", second_color="red"):
    """Draw a color flow that moves across display."""
    global current_colors

    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)

    first_rgb = get_rgb_from_color_name(first_color)
    second_rgb = get_rgb_from_color_name(second_color)

    for i in range(int(strip.numPixels() / 2)):
        wheel_pos = ((i * 256 // int(strip.numPixels() / 2)) + pos) % 256
        first, second = wheel(wheel_pos, reduce)

        # Calculate the target color
        first = int(first * 50 * mean_vol)
        second = int(second * 50 * mean_vol)
        divider = min(np.max(first_rgb) + np.max(second_rgb), 255)
        target_r = int((first_rgb[0] * first + second_rgb[0] * second) / divider)
        target_g = int((first_rgb[1] * first + second_rgb[1] * second) / divider)
        target_b = int((first_rgb[2] * first + second_rgb[2] * second) / divider)

        # Get the current color
        current_color = current_colors[i]

        # Smoothly transition to the target color
        new_color = smooth_transition(current_color, (target_r, target_g, target_b))

        # Clip values to 0-4095 range to prevent overflows
        r, g, b = [min(max(c, 0), 4095) for c in new_color]

        # Set the pixel color
        strip.setPixelColor(i, Color(r, g, b))

        # Update current color
        current_colors[i] = (r, g, b)

    strip.show()


def wheel(pos, reduce=2):
    """Generate color spectrum across 0-255 positions."""
    if pos < 128:
        return (pos * 2) // reduce, (255 - pos * 2) // reduce  # divide by 4 for reduced brightness
    else:
        pos -= 128
        return (255 - pos * 2) // reduce, (pos * 2) // reduce  # divide by 4 for reduced brightness


def led_strobe_effect(duration_seconds, frequency_ms):
    global strip

    end_time = time.time() + duration_seconds
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()

    while time.time() < end_time:
        # Alle LEDs auf Weiß setzen
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(220, 220, 255))
        strip.show()

        # Kurze Pause entsprechend der Frequenz
        time.sleep(frequency_ms / 1000.0)

        # Alle LEDs ausschalten
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()

        # Kurze Pause entsprechend der Frequenz
        time.sleep(frequency_ms / 1000.0)


def lin_lerp(a, b, t):
    """Linear interpolation between a and b"""
    return int(a * (1 - t) + b * t)


def lerp(a, b, t, exponent=3):
    """Exponential interpolation between a and b"""
    return int(a * ((1 - t) ** exponent) + b * (t ** exponent))


def adjust_brightness(color, factor):
    """ Passt die Helligkeit einer Farbe an. """
    return int(color * factor)


@led_in_thread
def led_music_visualizer(data, first_color="blue", second_color="red"):
    global last_show_time, global_led_state

    first_r, first_g, first_b = get_rgb_from_color_name(first_color)
    second_r, second_g, second_b = get_rgb_from_color_name(second_color)

    num_leds = strip.numPixels()
    num_leds_front = int(num_leds / 2)
    mid_point = int(num_leds / 2) // 2
    data = int(data * mid_point)

    # Initialisiere led_array mit halben Werten von global_led_state
    led_array = [(int(r / 2.5), int(g / 2.5), int(b / 2.5)) for r, g, b in global_led_state]
    global_led_state = led_array

    for pos in range(data):

        t = pos / mid_point
        brightness_factor = 0.75

        # Berechnung der neuen Farbwerte mit Helligkeitsanpassung
        bright_r = adjust_brightness(lerp(first_r, second_r, t), brightness_factor)
        bright_g = adjust_brightness(lerp(first_g, second_g, t), brightness_factor)
        bright_b = adjust_brightness(lerp(first_b, second_b, t), brightness_factor)

        # Speichern der Farben im Array statt direktem Setzen
        led_array[mid_point - pos] = (bright_r, bright_g, bright_b)
        led_array[mid_point + pos] = (bright_r, bright_g, bright_b)
        led_array[num_leds_front + mid_point - pos] = (
            lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t))
        led_array[num_leds_front + mid_point + pos] = (
            lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t))

    # Aktualisieren des LED-Streifens basierend auf dem Array
    for i, color in enumerate(led_array):
        strip.setPixelColor(i, Color(*color))

    strip.show()
