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

# import cProfile
from rpi_ws281x import *
from collections import deque
import time
import random
import pyaudio
import numpy as np
from helpers import *
from eurolite_t36 import *

# Global variable for saving the timestamp of the last strip.show() call
last_show_time = 0

# LED-strip configuration:
LED_COUNT = 270  # Anzahl der LED-Pixel.
LED_PIN = 18  # GPIO-Pin, der mit dem Datenpin des LED-Streifens verbunden ist.
LED_FREQ_HZ = 800000  # LED-Signalfrequenz in Hz (normalerweise 800khz)
LED_DMA = 10  # DMA-Kanal, der für die Signalgenerierung verwendet wird (kann 0-14 sein).
LED_BRIGHTNESS = 255  # Eingestellte Helligkeit (0 für dunkelste und 255 für hellste)
LED_INVERT = False  # True zum Invertieren des Signals (bei Verwendung von NPN-Transistor-Level-Shifter)
LED_CHANNEL = 0  # set to '1' for GPIOs 13, 19, 41, 45 or 53

# Create NeoPixel object with appropriate configuration.
strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
# Intialize the library (must be called once before other functions).
strip.begin()
current_time = 0
global_led_state = [(0, 0, 0) for _ in range(strip.numPixels())]

recent_audio_inputs = deque(maxlen=int(3))  # adjust as needed, originally 20


def star_chase(c, wait_ms):
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


def color_wipe(color, wait_ms=50):
    """Füllen Sie den Streifen nacheinander mit einer Farbe aus. Wartezeit in ms zwischen den Pixeln."""
    for i in range(int(strip.numPixels())):
        strip.setPixelColor(i, color)
        if wait_ms != 0:
            strip.show()
            time.sleep(wait_ms / 1000.0)
    if wait_ms == 0:
        strip.show()


def set_all_pixels_color(red, green, blue):
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


def color_flow_old(pos, audio_input, reduce=2):
    """Draw a color flow that moves across display."""
    """Old version. Can be deleted in later revisions."""
    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)
    # print(mean_vol)

    for i in range(int(strip.numPixels() / 2)):
        # tricky math! we use each pixel as a fraction of the full 96-color wheel
        # (int(strip.numPixels() / 2) steps) % 96 to make the wheel progress
        wheel_pos = ((i * 256 // int(strip.numPixels() / 2)) + pos) % 256
        r, b = wheel(wheel_pos, reduce)

        # Clip values to 0-4095 range to prevent overflows
        r = min(max(int(r * 50 * mean_vol), 0), 4095)
        b = min(max(int(b * 50 * mean_vol), 0), 4095)

        # Set the pixel color
        strip.setPixelColor(i, Color(r, 0, b))
    strip.show()


# Global variable to keep track of current LED colors
current_colors = [(0, 0, 0)] * strip.numPixels()


def smooth_transition(current, target, step=5):
    """Smoothly transition from current color to target color."""
    r = int(current[0] + min(max(target[0] - current[0], -step), step))
    g = int(current[1] + min(max(target[1] - current[1], -step), step))
    b = int(current[2] + min(max(target[2] - current[2], -step), step))
    return r, g, b


def color_flow(pos, audio_input, reduce=2, first_color="blue", second_color="red"):
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


# Rot einfärben
# color_wipe(Color(255, 0, 0))

# star_chase(Color(127, 127, 127), 50)
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
    global last_show_time, current_time, global_led_state

    first_r, first_g, first_b = get_rgb_from_color_name(first_color)
    second_r, second_g, second_b = get_rgb_from_color_name(second_color)

    num_leds = strip.numPixels()
    # Erstellen eines Arrays zur Speicherung der Farbinformationen für jede LED
    led_array = [(0, 0, 0) for _ in range(num_leds)]  # Startet mit allen LEDs ausgeschaltet

    num_leds_front = int(num_leds / 2)
    mid_point = int(num_leds / 2) // 2
    data = int(data * mid_point)

    # Initialisiere led_array mit halben Werten von global_led_state
    led_array = [(int(r / 2.5), int(g / 2.5), int(b / 2.5)) for r, g, b in global_led_state]
    global_led_state = led_array

    for pos in range(data):
        current_time = time.time()

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

    active_leds = sum(1 for color in led_array if color != (0, 0, 0))

    # Bedingungen für das Aktualisieren der Anzeige basierend auf der Anzahl der Datenpunkte (data)
    if active_leds < 15:
        # Wenn die Datenmenge sehr klein ist, aktualisiere bei jedem Durchgang
        strip.show()
        last_show_time = current_time
    elif active_leds < 30 and (active_leds - 1) % 2 == 0:
        # Für eine mäßige Datenmenge und wenn der letzte Datenpunkt auf eine gerade Zahl fällt, aktualisiere die Anzeige
        strip.show()
        last_show_time = current_time
    elif (active_leds - 1) % 5 == 0:
        # Für größere Datenmengen aktualisiere nur, wenn der letzte Datenpunkt ein Vielfaches von 5 ist
        strip.show()
        last_show_time = current_time
    else:
        # Wenn keine der Bedingungen zutrifft, führe eine Standardaktualisierung durch
        strip.show()

