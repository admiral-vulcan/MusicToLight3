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

# Global list to store active threads
active_threads = []


def calc_address(num):
    return (num * 6) - 6


def run_in_thread(function, args):
    """
    Run the provided function in a new thread.

    Arguments:
    function -- the function to run
    args -- tuple of arguments to pass to the function

    If a thread with the same function name is already running,
    no new thread is started.
    """
    global active_threads

    # Check if a thread with the desired function is already running
    for thread in active_threads:
        if thread.is_alive() and thread.name == function.__name__:
            # If thread is already running, don't start a new one
            return

    # If not, create and start a new thread
    thread = threading.Thread(target=function, args=args, name=function.__name__)
    thread.start()

    # Add the new thread to the list of active threads
    active_threads.append(thread)


def exponential_decrease(current_value, upper_limit):
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
