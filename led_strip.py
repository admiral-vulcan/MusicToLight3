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
    """
    A class that mimics the Adafruit NeoPixel library's functionality for use with UDP communication.
    This class is designed to be compatible with Adafruit's NeoPixel LED code but instead of controlling
    LEDs directly via hardware, it sends RGB values over UDP to an LED controller.

    Attributes:
        num_leds (int): Number of LEDs in the strip.
        colors (list of tuple): Current color data for each LED (stored as tuples of (R, G, B)).
        ip_address (str): IP address of the UDP receiver, which controls the actual LEDs.
        port (int): The UDP port number used for sending data.

    Methods:
        numPixels(): Returns the number of LEDs.
        setPixelColor(i, color): Sets the color of a specific LED.
        show(): Sends the current color data of all LEDs via UDP.
    """

    def __init__(self, num_leds):
        """
        Initializes the UDPNeoPixel instance with the specified number of LEDs.

        Args:
            num_leds (int): The total number of LEDs in the strip.
        """
        self.num_leds = num_leds
        self.colors = [(0, 0, 0)] * num_leds  # Initialize all LEDs to be turned off (black).
        self.ip_address = '192.168.1.153'  # Default IP address of the LED controller
        self.port = 4210  # Default port for the LED control commands

    def numPixels(self):
        """
        Returns the number of LEDs in the strip.

        Returns:
            int: The total number of LEDs.
        """
        return self.num_leds

    def setPixelColor(self, i, color):
        """
        Sets the color of a single LED by index.

        Args:
            i (int): The index of the LED to set.
            color (tuple or int): The color to set the LED to, either as an RGB tuple or a 24-bit integer.
        """
        if isinstance(color, tuple) and len(color) == 3:
            if 0 <= i < self.num_leds:
                self.colors[i] = color  # Set color from a tuple directly
        elif isinstance(color, int):
            # Convert the 24-bit color to an RGB tuple
            r = (color >> 16) & 0xFF
            g = (color >> 8) & 0xFF
            b = color & 0xFF
            self.colors[i] = (r, g, b)  # Update the color array

    def show(self):
        """
        Sends the color data for all LEDs over UDP as a binary message.
        This function constructs a binary message and sends it to the specified IP address and port.
        """
        message = bytearray(b'mls_')  # Prefix for the message to identify it
        message.extend(self.num_leds.to_bytes(2, byteorder='little'))  # Number of LEDs as two bytes

        for r, g, b in self.colors:
            # Ensure R, G, B values are within byte range
            message.extend([int(r) % 256, int(g) % 256, int(b) % 256])  # Append each color as one byte

        # Send the message over UDP to the controller
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


def led_star_chase(color, wait_ms):
    """
    Creates a 'star chase' effect on an LED strip, where stars (light points) chase across the strip.

    Args:
        color (int): The color value for the LEDs during the chase.
        wait_ms (int): The wait time in milliseconds between updates to the LED strip.

    Description:
        This function divides the LED strip into several segments and randomly lights up LEDs within those
        segments to simulate a 'chasing' effect. The effect iterates multiple times across the strip,
        and LEDs are turned on and off in sequence to create movement.

        Randomness is introduced in the start and stop positions within each segment, and also in the
        step size between lit LEDs, enhancing the dynamic and unpredictable nature of the effect.
        The function also includes a trailing off effect, where LEDs are turned off in a staggered manner.
    """
    num_chases = 30  # Total number of chase sequences to perform
    num_pixels_half = int(strip.numPixels() / 2)  # Operate on half of the pixels for the chase

    for chase_index in range(num_chases):
        # Calculate start position for the current chase, adjust with randomness
        start_position = (num_pixels_half / num_chases * chase_index) - random.randint(5, 15)
        if start_position < 0:
            start_position = 0

        # Calculate stop position for the current chase, adjust with randomness
        stop_position = (num_pixels_half / num_chases * (chase_index + 1)) + random.randint(5, 15)
        if stop_position > num_pixels_half:
            stop_position = num_pixels_half

        # Illuminate LEDs from start to stop position with an offset to create the chase effect
        for offset in range(3):
            position = start_position
            while position < stop_position:
                if 0 <= int(position + offset) < num_pixels_half:  # Ensure the position is within valid bounds
                    strip.setPixelColor(int(position + offset), color)
                position += random.randint(1, 15)  # Move to the next position randomly within 1 to 15 steps
            strip.show()  # Update the LED strip with new light positions

            time.sleep(wait_ms / 1000.0)  # Wait for the specified time

            # Turn off pixels to create a trailing effect
            for pixel_reset in range(0, num_pixels_half, 3):
                if 0 <= int(pixel_reset + offset) < num_pixels_half:
                    strip.setPixelColor(int(pixel_reset + offset), 0)

        # Final update to turn off all LEDs in this segment before the next chase starts
        strip.show()
        for pixel_reset in range(num_pixels_half):
            strip.setPixelColor(pixel_reset, 0)  # Turn off all LEDs
        strip.show()  # Apply the changes to the strip

    # Optionally control external devices like Eurolite T36
    set_eurolite_t36(5, 0, 0, 0, 255, 0)


def led_set_all_pixels_color(red, green, blue):
    """
    Sets the color of all pixels on an LED strip to a specified RGB value.

    Args:
        red (int): The red component of the color, in the range 0-255.
        green (int): The green component of the color, in the range 0-255.
        blue (int): The blue component of the color, in the range 0-255.

    Globals:
        strip (object): The LED strip object, which controls a series of LEDs.

    Description:
        This function takes RGB values and converts them into a single 32-bit integer color value.
        It then iterates over all pixels in the strip and sets each one to this color. After setting the colors,
        the strip is updated to reflect these changes visually. This method is particularly useful for creating
        a uniform color background or resetting the strip to a baseline color.
    """

    global strip
    # Convert the RGB values into a 32-bit integer color (format: 0xRRGGBB)
    color = (red << 16) | (green << 8) | blue

    # Iterate over each pixel in the strip and set its color
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)

    # Update the LED strip to display the new colors
    strip.show()


def smooth_transition(current, target, step=5):
    """
    Smoothly transitions from a current color to a target color by incrementally adjusting
    the color components (red, green, blue) towards the target within the specified step limit.

    Args:
        current (tuple): The current color as an RGB tuple (red, green, blue).
        target (tuple): The target color as an RGB tuple (red, green, blue).
        step (int): The maximum change allowed in any color component in a single call.

    Returns:
        tuple: An RGB tuple representing the new color after applying one step of the transition.

    Description:
        The function modifies each component (R, G, B) of the current color, moving it towards
        the target color without exceeding the defined step size. This helps in creating a
        gradual transition effect rather than an immediate shift, which is visually smoother.
        The step parameter controls the speed of the transition:
        - Smaller step values result in a slower and smoother transition.
        - Larger step values lead to a quicker but potentially more noticeable change.

        Each color component is adjusted independently. If the difference between the current and
        target component is greater than the step size, the component is increased or decreased by
        the step size. If the difference is less than the step size, it adjusts to the target value.
    """
    # Calculate the new red component
    r = int(current[0] + min(max(target[0] - current[0], -step), step))
    # Calculate the new green component
    g = int(current[1] + min(max(target[1] - current[1], -step), step))
    # Calculate the new blue component
    b = int(current[2] + min(max(target[2] - current[2], -step), step))

    return r, g, b


@led_in_thread
def led_color_flow(pos, audio_input, reduce=2, first_color="blue", second_color="red"):
    """
    Visualizes an audio-responsive color flow on an LED strip. The color flow's intensity and
    position are influenced by the volume of the audio input.

    Args:
        pos (int): Current position on the strip, used to calculate color offsets.
        audio_input (float): Current audio input level, used to modulate the color intensity.
        reduce (int): Factor to reduce the color intensity, helping in managing LED brightness.
        first_color (str): The name of the initial color in the flow.
        second_color (str): The name of the final color in the flow.

    Globals:
        current_colors (list): Stores the current color of each LED for smooth transitions.
        recent_audio_inputs (list): History of recent audio inputs to calculate a moving average.

    Decorators:
        @led_in_thread: Ensures that the LED updates do not block the main execution thread,
                        allowing smooth audio processing.

    Description:
        The function calculates a moving average of audio inputs to determine the overall volume
        level. It then uses this volume to modulate the intensity of colors generated by a
        color wheel mechanism, creating a dynamic, audio-responsive LED display.
    """
    global current_colors, recent_audio_inputs

    # Append current audio input to the history and compute the mean volume
    recent_audio_inputs.append(audio_input)
    mean_vol = safe_mean(recent_audio_inputs)

    # Get RGB values for the provided color names
    first_rgb = get_rgb_from_color_name(first_color)
    second_rgb = get_rgb_from_color_name(second_color)

    # Calculate color values for each LED based on the audio volume and position
    for i in range(int(strip.numPixels() / 2)):
        # Calculate position on color wheel
        wheel_pos = ((i * 256 // int(strip.numPixels() / 2)) + pos) % 256
        first, second = wheel(wheel_pos, reduce)

        # Modulate color intensity based on audio volume
        first = int(first * 50 * mean_vol)
        second = int(second * 50 * mean_vol)
        divider = min(np.max(first_rgb) + np.max(second_rgb), 255)

        # Calculate target colors by blending the two color ranges
        target_r = int((first_rgb[0] * first + second_rgb[0] * second) / divider)
        target_g = int((first_rgb[1] * first + second_rgb[1] * second) / divider)
        target_b = int((first_rgb[2] * first + second_rgb[2] * second) / divider)

        # Retrieve current color from global state for smooth transition
        current_color = current_colors[i]

        # Apply a smooth transition from current to target color
        new_color = smooth_transition(current_color, (target_r, target_g, target_b))

        # Clip the color values to ensure they are within the valid range for the LEDs
        r, g, b = [min(max(c, 0), 4095) for c in new_color]

        # Set the new color for the LED and update the global color state
        strip.setPixelColor(i, Color(r, g, b))
        current_colors[i] = (r, g, b)

    # Refresh the LED strip to apply the new color settings
    strip.show()


def wheel(pos, reduce=2):
    """
    Generates a part of the color wheel spectrum based on the position.
    This function is typically used for color cycling effects on RGB LEDs.

    Args:
        pos (int): Position on the color wheel (0-255) to generate the color.
        reduce (int): The factor by which to reduce the brightness of the colors.

    Returns:
        tuple: A tuple of two integers representing the RGB values. Only two values are returned,
               because this example simplifies the color wheel to two dimensions.

    Description:
        The function maps the input position to a simplified color spectrum where:
        - From 0 to 127, it linearly interpolates between red and blue.
        - From 128 to 255, it linearly interpolates between blue and red.
        The reduction factor decreases the intensity/brightness of the colors to prevent LED saturation.
    """
    if pos < 128:
        return (pos * 2) // reduce, (255 - pos * 2) // reduce
    else:
        pos -= 128
        return (255 - pos * 2) // reduce, (pos * 2) // reduce


def led_strobe_effect(duration_seconds, frequency_ms):
    """
    Creates a strobe light effect on an LED strip by alternating between all LEDs being on and off.

    Args:
        duration_seconds (float): The total duration of the strobe effect in seconds.
        frequency_ms (int): The frequency of the strobe effect in milliseconds (time between flashes).

    Globals:
        strip: The LED strip object controlled by this function.

    Description:
        The function continuously toggles all LEDs on the strip between white and off state,
        creating a flashing effect. The frequency determines how fast the LEDs toggle, and the
        duration specifies how long the effect should last.
    """
    global strip

    end_time = time.time() + duration_seconds
    # Turn off all LEDs initially
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()

    while time.time() < end_time:
        # Turn all LEDs to white
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(220, 220, 255))  # Slightly off-white color
        strip.show()

        # Pause for half the frequency duration
        time.sleep(frequency_ms / 1000.0)

        # Turn off all LEDs
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, Color(0, 0, 0))
        strip.show()

        # Pause for half the frequency duration
        time.sleep(frequency_ms / 1000.0)


def adjust_brightness(color, factor):
    """
    Adjusts the brightness of a color component by multiplying it with a factor.
    This function assumes that the color component and the factor are valid.

    Args:
        color (int): The color component (0-255).
        factor (float): The factor by which to adjust the brightness.

    Returns:
        int: The adjusted color component, clipped to the range 0-255.
    """
    # Multiply the color by the factor and clip to the max 255 value to avoid overflow
    return max(0, min(255, int(color * factor)))


@led_in_thread
def led_music_visualizer(data, first_color="blue", second_color="red"):
    """
    Visualizes music data on an LED strip by interpolating between two colors based on the music intensity.
    Uses global state to maintain continuity and fading effects between visualizations.

    Args:
        data (float): A value representing music intensity, scaled to affect the LED display.
        first_color (str): The name of the first color in the gradient.
        second_color (str): The name of the second color in the gradient.

    Globals:
        last_show_time (float): Timestamp of the last update to the LED strip.
        global_led_state (list): Current state of the LED colors, used for transition effects.

    Decorators:
        @led_in_thread: Ensures that LED updates do not block the main execution thread.
    """
    global last_show_time, global_led_state

    # Set the front brightness
    brightness_factor = 0.5

    # Set the overall fade rate for previous state
    pre_rate = 0.8

    # Convert color names to RGB values
    first_r, first_g, first_b = get_rgb_from_color_name(first_color)
    second_r, second_g, second_b = get_rgb_from_color_name(second_color)

    # Determine number of LEDs and calculate positions
    num_leds = strip.numPixels()
    num_leds_front = int(num_leds / 2)
    mid_point = num_leds_front // 2
    data = int(data * mid_point)

    # Initialize LED array with faded previous colors
    led_array = [(int(r * pre_rate), int(g * pre_rate), int(b * pre_rate)) for r, g, b in global_led_state]
    global_led_state = led_array

    # Calculate new colors based on music data and set them in the LED array
    for pos in range(data):
        t = pos / mid_point  # Calculate interpolation factor

        # Interpolate colors and adjust brightness
        bright_r = adjust_brightness(exp_lerp(first_r, second_r, t), brightness_factor)
        bright_g = adjust_brightness(exp_lerp(first_g, second_g, t), brightness_factor)
        bright_b = adjust_brightness(exp_lerp(first_b, second_b, t), brightness_factor)

        # Set colors symmetrically around the midpoint
        led_array[mid_point - pos] = (bright_r, bright_g, bright_b)
        led_array[mid_point + pos] = (bright_r, bright_g, bright_b)
        led_array[num_leds_front + mid_point - pos] = (
            exp_lerp(first_r, second_r, t), exp_lerp(first_g, second_g, t), exp_lerp(first_b, second_b, t))
        led_array[num_leds_front + mid_point + pos] = (
            exp_lerp(first_r, second_r, t), exp_lerp(first_g, second_g, t), exp_lerp(first_b, second_b, t))

    # Update the LED strip based on the prepared array
    for i, color in enumerate(led_array):
        strip.setPixelColor(i, Color(*color))

    strip.show()
