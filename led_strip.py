from rpi_ws281x import *
from collections import deque
import time
import random
import pyaudio
import numpy as np
from helpers import *

# LED-Streifen Konfiguration:
LED_COUNT = 180  # Anzahl der LED-Pixel.
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

recent_audio_inputs = deque(maxlen=int(50))  # adjust as needed


def theater_chase(c, wait_ms):
    NUMCHASE = 30
    NUMPIX = LED_COUNT
    for j in range(NUMCHASE):
        startPo = (NUMPIX / NUMCHASE * j) - random.randint(5, 15)
        if startPo < 0:
            startPo = 0
        stopPo = (NUMPIX / NUMCHASE * (j + 1)) + random.randint(5, 15)

        for q in range(3):
            i = startPo
            while i < stopPo:
                strip.setPixelColor(int(i + q), c)
                i += random.randint(1, 15)
            strip.show()

            time.sleep(wait_ms / 1000.0)

            for i in range(0, NUMPIX, 3):
                strip.setPixelColor(int(i + q), 0)

        strip.show()  # Sende die Farbänderungen an den Streifen
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, 0)  # Schalte alle LEDs aus
        strip.show()  # Sende die Farbänderungen an den Streifen


# Define function to visualize music on LED strip
def music_visualizer(audio_input):
    recent_audio_inputs.append(audio_input)
    mean_vol = np.mean(recent_audio_inputs)
    if mean_vol > 0:
        to_mean_factor = 1/mean_vol
    else:
        to_mean_factor = 1

    # audio_input *= to_max_factor

    # Convert audio input to an integer in the range 0 to strip.numPixels() / 2
    num_leds = int(to_mean_factor * audio_input * (strip.numPixels() // 2))
    num_leds = exponential_decrease(num_leds, 128)
    # print(num_leds)

    mid_point = strip.numPixels() // 2

    # Visualize the audio energy on the LED strip
    for i in range(strip.numPixels()):
        if mid_point - num_leds <= i <= mid_point + num_leds:
            # Set the color of lit LEDs based on the audio energy
            # feel free to adjust colors and the color scaling
            red = int(exponential_decrease(mean_vol * 255, 255))
            green = int(exponential_decrease(mean_vol * 255, 255))
            blue = int(exponential_decrease(mean_vol * 255, 255))
            strip.setPixelColor(i, Color(red, green, blue))
        else:
            # Turn off the rest of the LEDs
            strip.setPixelColor(i, Color(0, 0, 0))

    # Update the strip
    strip.show()


def color_wipe(color, wait_ms=50):
    """Füllen Sie den Streifen nacheinander mit einer Farbe aus. Wartezeit in ms zwischen den Pixeln."""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
        if wait_ms != 0:
            strip.show()
            time.sleep(wait_ms / 1000.0)
    if wait_ms == 0:
        strip.show()


# Rot einfärben
# color_wipe(Color(255, 0, 0))

# theater_chase(Color(127, 127, 127), 50)

"""
while True:
    # Read some data from audio input
    audiobuffer = line_in.read(int(buffer_size / 2), exception_on_overflow=False)
    signal_input = np.frombuffer(audiobuffer, dtype=np.float32)

    # Calculate LED strip visualization based on audio energy
    music_visualizer(np.max(signal_input))

    time.sleep(0.05)
"""
