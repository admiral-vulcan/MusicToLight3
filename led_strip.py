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

# LED-Streifen Konfiguration:
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


# Define function to visualize music on LED strip
@led_in_thread
def led_music_visualizer_old(audio_input):
    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)
    if mean_vol > 0:
        to_mean_factor = 1 / mean_vol
    else:
        to_mean_factor = 1

    # Convert audio input to an integer in the range 0 to int(strip.numPixels() / 2) / 2
    num_leds = int(to_mean_factor * audio_input * (int(strip.numPixels() / 2) // 2))
    num_leds = exponential_decrease(num_leds, 128)

    mid_point = int(strip.numPixels() / 2) // 2

    # Cap mean_vol at 1
    mean_vol = min(mean_vol, 1)

    # Visualize the audio energy on the LED strip
    for i in range(int(strip.numPixels() / 2)):
        # Calculate the distance from the mid point in range [0, 1]
        distance_from_mid = abs(i - mid_point) / (int(strip.numPixels() / 2) / 2)
        distance_from_mid = min(distance_from_mid, 1)  # Ensure distance_from_mid is not more than 1
        # Exponential increase in red color as we move away from the center
        red = int((distance_from_mid ** 2) * mean_vol * 255)
        red = min(red, 255)  # Ensure red is not more than 255
        # Exponential decrease in blue color as we move away from the center
        blue = int(((1 - distance_from_mid) ** 2) * mean_vol * 255)
        blue = min(blue, 255)  # Ensure blue is not more than 255
        # print(f"mean_vol: {mean_vol}, distance_from_mid: {distance_from_mid}, red: {red}, blue: {blue}")

        if mid_point - num_leds <= i <= mid_point + num_leds:
            strip.setPixelColor(i, Color(red, 0, blue))
        else:
            # Turn off the rest of the LEDs
            strip.setPixelColor(i, Color(0, 0, 0))

    # Update the strip
    strip.show()


@led_in_thread
def led_music_visualizer_ols(audio_input, first_color="blue", second_color="red"):
    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)
    if mean_vol > 0:
        to_mean_factor = 1 / mean_vol
    else:
        to_mean_factor = 1

    first_rgb = get_rgb_from_color_name(first_color)
    second_rgb = get_rgb_from_color_name(second_color)

    num_leds = int(to_mean_factor * audio_input * (int(strip.numPixels() / 2) // 2))
    num_leds = exponential_decrease(num_leds, 128)

    mid_point = int(strip.numPixels() / 2) // 2
    mean_vol = min(mean_vol, 1)

    for i in range(int(strip.numPixels() / 2)):
        distance_from_mid = abs(i - mid_point) / (int(strip.numPixels() / 2) / 2)
        distance_from_mid = min(distance_from_mid, 1)

        # Calculate the new RGB values based on the ratios
        r = int((second_rgb[0] * (distance_from_mid ** 2) + first_rgb[0] * ((1 - distance_from_mid) ** 2)) * mean_vol)
        g = int((second_rgb[1] * (distance_from_mid ** 2) + first_rgb[1] * ((1 - distance_from_mid) ** 2)) * mean_vol)
        b = int((second_rgb[2] * (distance_from_mid ** 2) + first_rgb[2] * ((1 - distance_from_mid) ** 2)) * mean_vol)

        # Clip values to 0-4095 range to prevent overflows
        r = min(max(r, 0), 4095)
        g = min(max(g, 0), 4095)
        b = min(max(b, 0), 4095)

        if mid_point - num_leds <= i <= mid_point + num_leds:
            strip.setPixelColor(i, Color(r, g, b))
        else:
            strip.setPixelColor(i, Color(0, 0, 0))

    strip.show()


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


def color_flow(pos, audio_input, reduce=2, first_color="blue", second_color="red"):
    """Draw a color flow that moves across display."""
    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)

    first_rgb = get_rgb_from_color_name(first_color)
    second_rgb = get_rgb_from_color_name(second_color)

    for i in range(int(strip.numPixels() / 2)):
        wheel_pos = ((i * 256 // int(strip.numPixels() / 2)) + pos) % 256
        first, second = wheel(wheel_pos, reduce)

        # Calculate first and second color
        first = (int(first * 50 * mean_vol))
        second = (int(second * 50 * mean_vol))

        # Find divider
        divider = min(np.max(first_rgb) + np.max(second_rgb), 255)

        # Calculate the new RGB values based on the ratios
        r = int((first_rgb[0] * first + second_rgb[0] * second) / divider)
        g = int((first_rgb[1] * first + second_rgb[1] * second) / divider)
        b = int((first_rgb[2] * first + second_rgb[2] * second) / divider)

        # Clip values to 0-4095 range to prevent overflows
        r = min(max(r, 0), 4095)
        g = min(max(g, 0), 4095)
        b = min(max(b, 0), 4095)

        # Set the pixel color
        strip.setPixelColor(i, Color(r, g, b))

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


@led_in_thread
def led_music_visualizer(data, first_color="blue", second_color="red"):
    set_all_pixels_color(0, 0, 0)
    first_r, first_g, first_b = get_rgb_from_color_name(first_color)
    second_r, second_g, second_b = get_rgb_from_color_name(second_color)

    num_leds = strip.numPixels()
    num_leds_front = int(num_leds / 2)
    mid_point = int(num_leds / 2) // 2
    data = int(data * mid_point)
    print(num_leds_front)

    for pos in range(data):
        t = pos / mid_point

        # front side
        strip.setPixelColor(mid_point - pos,
                            Color(lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t)))
        strip.setPixelColor(mid_point + pos,
                            Color(lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t)))

        # back side
        strip.setPixelColor(num_leds_front + mid_point - pos,
                            Color(lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t)))
        strip.setPixelColor(num_leds_front + mid_point + pos,
                            Color(lerp(first_r, second_r, t), lerp(first_g, second_g, t), lerp(first_b, second_b, t)))

        # animate a bit for smoothness
        if pos % 10 == 0:
            strip.show()
    strip.show()


"""
test code

while True:
    led_music_visualizer_new(1)
    time.sleep(0.1)
    led_music_visualizer_new(0.1)
    time.sleep(0.1)
    led_music_visualizer_new(0.3)
    time.sleep(0.1)
"""
