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

import RPi.GPIO as GPIO
import pyaudio
import numpy as np
import math
import threading
import time
import ctypes
from queue import Queue

# Global list to store active scanner threads
active_scan_threads = []

# Global variable to store the current HDMI thread
current_hdmi_thread = None

# Global variable to store the current LED thread
current_led_thread = None

# Global variable to store the timestamp of the last run of the HDMI thread
last_run_time = None

# Set the GPIO mode to BCM and disable warnings
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Configure GPIO 23 as an output (For Fog Machine)
GPIO.setup(23, GPIO.OUT)

# Filter configurations
thx_band = (1, 120)
low_band = (1, 300)
mid_band = (300, 2000)
high_band = (2000, 16000)

# PyAudio settings
buffer_size = 1024
hop_size = buffer_size // 2
pyaudio_format = pyaudio.paFloat32
n_channels = 1
sample_rate = 44100
device_index = 0


def calc_address(num):
    """
    Calculate and return the address offset based on the provided number.

    Arguments:
    num -- the number used to calculate the address offset
    """
    return (num * 6) - 6


def map_value(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another.

    Arguments:
    value -- the value to be mapped
    in_min, in_max -- the range of the input value
    out_min, out_max -- the range of the output value
    """
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def hdmi_in_thread(func):
    """
    Decorator to run the given function in a separate HDMI thread.

    Arguments:
    func -- the function to be executed in a thread

    This decorator ensures that the function is not executed more often than once every 0.2 seconds.
    It also checks if the current HDMI thread is alive and skips the function execution if it is.
    """
    global current_hdmi_thread
    global last_run_time

    def wrapped_function(*args, **kwargs):
        global current_hdmi_thread
        global last_run_time

        current_time = time.time()

        # Check if the function is called too frequently
        if last_run_time and current_time - last_run_time < 0.2:  # lesser is more stressful
            return

        # Check if the current HDMI thread is running
        if current_hdmi_thread and current_hdmi_thread.is_alive():
            return

        # Execute the function in a new thread
        current_hdmi_thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        current_hdmi_thread.start()

        # Update the last execution time
        last_run_time = time.time()

    return wrapped_function


def led_in_thread(func):
    """
    Decorator to run the given function in a separate LED thread.

    Arguments:
    func -- the function to be executed in a thread

    This decorator checks if the current LED thread is alive and skips the function execution if it is.
    """
    global current_led_thread

    def wrapped_function(*args, **kwargs):
        global current_led_thread

        # Check if the current LED thread is running
        if current_led_thread and current_led_thread.is_alive():
            return

        # Execute the function in a new thread
        current_led_thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        current_led_thread.start()

    return wrapped_function


def scan_in_thread(function, args):
    """
    Run the provided function in a new thread.

    Arguments:
    function -- the function to run
    args -- tuple of arguments to pass to the function

    If a thread with the same function name and the first argument is already running,
    no new thread is started. The first argument is considered only if it's an integer between 0 and 255.
    """
    global active_scan_threads

    # Check if the first argument is an integer between 0 and 255
    arg_str = ''
    if args and isinstance(args[0], int) and 0 <= args[0] <= 255:
        arg_str = f"-{args[0]}"

    # Construct the thread name
    thread_name = f"{function.__name__}{arg_str}"

    # Check if a thread with the desired function and the first argument is already running
    for thread in active_scan_threads:
        if thread.is_alive() and thread.name == thread_name:
            # If thread is already running, don't start a new one
            return

    # If not, create and start a new thread
    thread = threading.Thread(target=function, args=args, name=thread_name)
    thread.start()

    # Add the new thread to the list of active threads
    active_scan_threads.append(thread)


def exponential_decrease(current_value, upper_limit=255):
    """
    Decrease the provided value exponentially.

    Arguments:
    current_value -- the current value
    upper_limit -- the maximum allowed value

    The function returns an exponentially decreased value. If the decreased value
    would be above the upper limit, the upper limit is returned instead.
    """
    k = -math.log(upper_limit) / upper_limit
    new_value = math.exp(-k * current_value)
    return min(new_value, upper_limit)


def invert(current_value, upper_limit):
    """
    Invert the provided value within the upper limit.

    Arguments:
    current_value -- the current value
    upper_limit -- the maximum allowed value

    The function returns an inverted value. If the inverted value
    would be above the upper limit, the upper limit is returned instead.
    """
    new_value = (current_value * -1) + upper_limit
    return min(new_value, upper_limit)


def safe_mean(array):
    """
    Calculate the mean of a list or numpy array, safely handling empty inputs.
    """
    if len(array) > 0:
        return np.mean(array)
    else:
        return 0


def compute_mean_volume(volumes):
    return sum(volumes) / len(volumes)


def reduce_signal(signal, target_length=15):
    step_size = len(signal) // target_length
    reduced_signal = [safe_mean(signal[i:i + step_size]) for i in range(0, len(signal), step_size)]
    return reduced_signal[:target_length]


def generate_matrix(low_signal, mid_signal, high_signal, low_mean, mid_mean, high_mean):
    matrix = []

    # Werte unter 0,01 zu 0 ändern und auf 2 Nachkommastellen runden
    def process_signal(signal):
        return [0 if value < 0.01 else round(value, 2) for value in signal]

    low_signal = process_signal(low_signal)
    mid_signal = process_signal(mid_signal)
    high_signal = process_signal(high_signal)

    # Für jedes Signal: reduziere auf 15 Werte und bestimme für jeden Wert, ob er über (1) oder unter (0) dem Mittelwert liegt.
    for signal, mean in [(low_signal, low_mean), (mid_signal, mid_mean), (high_signal, high_mean)]:
        reduced_signal = reduce_signal(signal)
        for i in range(3):  # für jede der drei Reihen für ein bestimmtes Signal
            if i == 0:  # obere Zeile
                value_booster = 2.5
            elif i == 1:  # mittlere Zeile
                value_booster = 1.25
            else:  # untere Zeile
                value_booster = 0.5

            matrix.append([1 if (value * value_booster) > mean else 0 for value in reduced_signal])

    return matrix


def terminate_thread(thread):
    if not thread:
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)

    # if res == 0:
    #    raise ValueError("Nonexistent thread id")
    if res != 1:
        # Wenn es Probleme gibt, setze es zurück
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        # raise SystemError("Failed to forcefully kill the thread")


def kill_current_hdmi():
    terminate_thread(current_hdmi_thread)


def string_to_int(s):
    """Konvertiert einen String in eine Ganzzahl."""
    result = 0
    for char in s:
        result = (result << 8) | ord(char)
    return result


def interpret_color(color):
    """
    Interpret a given RGB color and return the closest known color name.
    :param color: A tuple of RGB values (R, G, B)
    :return: A string representing the color name
    """
    r, g, b = color

    # RGB definitions for known colors
    colors = {
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "green": (0, 128, 0),
        "orange": (255, 165, 0),
        "blue": (0, 0, 255),
        "pink": (255, 192, 203)
    }

    # Find the closest known color
    min_distance = float('inf')
    closest_color = None

    for color_name, rgb in colors.items():
        distance = sum([(a - b) ** 2 for a, b in zip(color, rgb)])  # Euclidean distance in RGB space
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color


def laser_color_to_int(color):
    """
    Convert a color string to an integer value based on a predefined table.

    Args:
    color (str): The color to convert.

    Returns:
    int or None: The integer value of the color, or None if the color is not in the table.
    """
    color_table = {
        "red": 10,
        "yellow": 20,
        "green": 25,
        "blue": 45,
        "purple": 50,
        "white": 60,
        # not supported but... let's deal with them:
        "orange": 20,  # like yellow
        "pink": 50  # like purple
    }

    return color_table.get(color.lower())


def get_rgb_from_color_name(color_name):
    """Get RGB tuple from color name."""
    colors = {
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "green": (0, 128, 0),
        "orange": (255, 165, 0),
        "blue": (0, 0, 255),
        "pink": (255, 192, 203)
    }
    return colors.get(color_name.lower(), (255, 255, 255))  # Default to white if color name is not found


def drop_color(drop_sum, red, green, blue):
    """
    Adjust the color based on the drop_sum.

    Args:
    - drop_sum: An integer between 1 and 32 inclusive.
    - red, green, blue: Integers representing color values.

    Returns:
    - A tuple (new_red, new_green, new_blue) representing the adjusted color.
    """

    # Erweitere die Range der drop_sum
    expanded_red = int(drop_sum * (red / 32))
    expanded_green = int(drop_sum * (green / 32))
    expanded_blue = int(drop_sum * (blue / 32))

    # Subtrahiere die erweiterte drop_sum von red, green und blue
    new_red = max(0, red - expanded_red)
    new_green = max(0, green - expanded_green)
    new_blue = max(0, blue - expanded_blue)

    return new_red, new_green, new_blue


# Perform FFT with zero padding to improve frequency resolution
def perform_fft_with_zero_padding(signal, sample_rate, zero_padding_factor=2):
    """Perform Fast Fourier Transform (FFT) with zero padding."""

    # Pad the signal for better frequency resolution
    padded_signal = np.pad(signal, (0, len(signal) * (zero_padding_factor - 1)), 'constant')

    # Perform FFT and split into magnitude and frequencies
    fft_output = np.fft.fft(padded_signal)
    fft_magnitude = np.abs(fft_output)[:len(fft_output) // 2]
    fft_frequencies = np.fft.fftfreq(len(fft_output), 1 / sample_rate)[:len(fft_output) // 2]

    return fft_magnitude, fft_frequencies


# Psychoacoustic analysis
def equal_loudness_curve(frequency):
    """Calculate the equal loudness curve weight based on a simplified ISO 226 model."""

    # ISO 226:2003 frequency and curve points
    freq_points = np.array([20, 100, 500, 1000, 5000, 10000, 20000])
    curve_points = np.array([29.8, 30.4, 33.0, 35.2, 41.5, 44.4, 45.4])

    # Interpolate to find the weight at the given frequency
    weight = np.interp(frequency, freq_points, curve_points)

    return weight


def apply_psycho_loudness_curve(fft_magnitude, fft_frequencies):
    """Apply the psychoacoustic loudness curve to the FFT magnitudes."""

    # Calculate the weight for each frequency and apply it
    weights = np.array([equal_loudness_curve(f) for f in fft_frequencies])
    weighted_fft_magnitude = fft_magnitude * weights

    return weighted_fft_magnitude


# Thread management for finding dominant harmonies
def harmony_thread(func):
    """Decorator to manage thread execution of the function."""

    func.thread = None
    func.queue = Queue()

    def wrapper(*args, **kwargs):
        # Start a new thread if none exists or the existing one has completed
        if func.thread is None or not func.thread.is_alive():
            func.thread = threading.Thread(target=func, args=(func.queue, *args), kwargs=kwargs)
            func.thread.start()

        # Retrieve and return result from queue if available
        if not func.queue.empty():
            return func.queue.get()

    return wrapper


@harmony_thread
def find_dominant_harmony_in_timeframe(queue, signal, sample_rate):
    """Find and return the dominant frequency in a given timeframe."""

    # Perform FFT and apply psychoacoustic weighting
    fft_magnitude, fft_frequencies = perform_fft_with_zero_padding(signal, sample_rate, 2)
    weighted_fft_magnitude = apply_psycho_loudness_curve(fft_magnitude, fft_frequencies)

    # Filter out frequencies below 20 Hz
    valid_indices = np.where(fft_frequencies >= 20)

    # Find the dominant frequency within the valid range
    dominant_harmony = int(fft_frequencies[valid_indices][np.argmax(weighted_fft_magnitude[valid_indices])])

    # Put the result into the queue
    queue.put(dominant_harmony)
