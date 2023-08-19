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

import numpy as np
import math
import threading
import ctypes

# Global list to store active scanner threads
active_scan_threads = []

# Globale Variable zum Speichern des aktuellen hdmi Threads
current_hdmi_thread = None

# Globale Variable zum Speichern des aktuellen hdmi Threads
current_led_thread = None


def calc_address(num):
    return (num * 6) - 6


def map_value(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another.
    """
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def hdmi_in_thread(func):
    global current_hdmi_thread

    def wrapped_function(*args, **kwargs):
        global current_hdmi_thread

        # Überprüfe, ob der aktuelle Thread läuft
        if current_hdmi_thread and current_hdmi_thread.is_alive():
            # print(f"{func.__name__} wurde übersprungen, da bereits ein Thread läuft.")
            return

        # Funktion in einem neuen Thread ausführen
        current_hdmi_thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        current_hdmi_thread.start()

    return wrapped_function


def led_in_thread(func):
    global current_led_thread

    def wrapped_function(*args, **kwargs):
        global current_led_thread

        # Überprüfe, ob der aktuelle Thread läuft
        if current_led_thread and current_led_thread.is_alive():
            # print(f"{func.__name__} wurde übersprungen, da bereits ein Thread läuft.")
            return

        # Funktion in einem neuen Thread ausführen
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
    if np.size(array) > 0:
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
    return int.from_bytes(s.encode(), 'big')


def int_to_string(i):
    """Konvertiert eine Ganzzahl zurück in einen String."""
    return i.to_bytes((i.bit_length() + 7) // 8, 'big').decode()
