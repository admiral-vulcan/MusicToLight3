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


import os
import pyaudio
import collections
import math
import numpy as np
import aubio
from scipy.signal import ellip, sosfilt, sos2zpk, lfilter_zi
from helpers import *

os.environ['ALSA_LOG_LEVEL'] = 'error'


def get_second_highest(values):
    """
    This function returns the second highest value in the given array.

    Args:
    values : A NumPy array.

    Returns:
    second_highest : Second highest value in the array.
    """
    if len(values) > 1:
        # Use numpy partition to rearrange the array
        values_partitioned = np.partition(values, -2)

        # The second highest value is now at index -2
        second_highest = values_partitioned[-2]

        return second_highest
    else:
        return values[0]


def adjust_gain_old(volumes, signal, target_volume=0.1, max_gain=5):
    average_volume = safe_mean(volumes)
    gain_factor = target_volume / average_volume if average_volume > 0 else 1.0
    if gain_factor > max_gain:
        gain_factor = max_gain
    if signal.any() > 0:
        print(signal)
    return signal * gain_factor, gain_factor


def adjust_gain(volumes, signal, target_volume=0.1, max_gain=5):
    # Berechne den durchschnittlichen Lautstärkepegel
    average_volume = safe_mean(volumes)

    # Berechne den Verstärkungsfaktor
    gain_factor = target_volume / average_volume if average_volume > 0 else 1.0

    # Begrenze den Verstärkungsfaktor auf den maximal zulässigen Wert
    if gain_factor > max_gain:
        gain_factor = max_gain

    # Setze Werte mit einem Betrag kleiner als 0,01 auf 0
    signal = signal * 1
    signal[np.abs(signal) < 0.01] = 0

    # Wende den Verstärkungsfaktor auf das Signal an
    adjusted_signal = signal * gain_factor

    return adjusted_signal, gain_factor


def design_filter(lowcut, highcut, sample_rate, ripple_db=0.5, stop_atten_db=40, order=3):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = ellip(order, ripple_db, stop_atten_db, [low, high], btype='band', output='sos')
    zi = np.zeros((sos.shape[0], 2))
    return sos, zi


def is_heavy(signal, delta_values, count_over, max_values, last_counts, heavy_counter):
    if len(last_counts) == 5 and max(last_counts) - min(last_counts) >= 3:
        heavy_counter = 2
    return np.max(np.abs(signal)) > 0.05 and (count_over > 3 or heavy_counter > 0) and np.max(
        delta_values) > 0.3, heavy_counter


def calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter):
    """
    Calculate a heaviness score between 0 and 10 based on the given parameters.
    """
    # normalize each parameter to a range of 0 to 1
    normalized_delta = min(delta_value / 0.5, 1)
    normalized_count = min(count_over / 10.0, 1)
    normalized_gain = min(gain_factor / 5.0, 1)
    normalized_heavy_counter = min(heavy_counter / 5.0, 1)

    # calculate heaviness as a weighted sum of parameters
    heaviness = 0.4 * normalized_delta + 0.3 * normalized_count + 0.2 * normalized_gain + 0.1 * normalized_heavy_counter

    # scale to a range of 0 to 10
    heaviness *= 10

    return heaviness
    # return exponential_decrease(heaviness, 10)


def dominant_frequency(signal, sample_rate):
    # Zentrierung des Signals
    signal_centered = signal - safe_mean(signal)

    # Überprüfung, ob das Signal nahe Null ist
    if np.max(np.abs(signal_centered)) < 0.01:  # 0.01 ist ein kleiner Schwellenwert, den Sie anpassen können
        return 0

    # Fourier-Transformation des zentrierten Signals
    fft_result = np.fft.rfft(signal_centered)

    # Absolutwerte der komplexen FFT-Ergebnisse, um die Amplitude jeder Frequenz zu erhalten
    amplitudes = np.abs(fft_result)

    # Bestimmung der Frequenz mit der höchsten Amplitude
    dominant_frequency_index = np.argmax(amplitudes)

    # Umwandlung des Indexes in die tatsächliche Frequenz
    dominant_frequency_hz = dominant_frequency_index * sample_rate / len(signal_centered)

    return dominant_frequency_hz


def detect_drop(volume_mean, heavy, dominant_frequencies, heaviness_history, drop_history, drop_length=50,
                rise_threshold=15, volume_rise_threshold=0.08, dominant_frequencies_threshold=300, heavy_threshold=0.8,
                drop_threshold=1, heavy_return_threshold=2):
    # Check that we have enough data
    if len(dominant_frequencies) < drop_length or len(heaviness_history) < drop_length:
        # print("Not enough data. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    if volume_mean > volume_rise_threshold:
        rise_threshold *= 1.5

    # If there have been enough drops, stay in drop state until it's over
    if sum(drop_history) >= drop_threshold and not heavy:
        # print("In drop state due to previous drops. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return True

    if sum(drop_history) > 0 and not heavy:
        return True

    # Get the dominant frequencies during the drop
    drop_dominant_frequencies = list(dominant_frequencies)[-drop_length:]

    # Get the heaviness history
    heavy_history = list(heaviness_history)[-drop_length:]

    # Check if the drop is over: heavy is True twice in a row
    if heavy_history[-heavy_return_threshold:] == [True] * heavy_return_threshold:
        # print("No drop/drop ended. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    # Check for the start of the drop: heavy becomes False and there was at least one True in heavy_history
    if heavy or True not in heavy_history:
        # print("No transition from heavy to non-heavy. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    # Check that there were enough heavies before the drop
    if sum(heavy_history) < heavy_threshold:
        # print("Not enough heaviness before the drop. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    # Check that the average dominant frequency during the drop is above 400 Hz/dominant_frequencies_threshold
    if safe_mean(drop_dominant_frequencies) <= dominant_frequencies_threshold:
        # print("Average dominant frequency during the drop is below 350 Hz. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    # Calculate the rise in dominant frequencies
    frequency_rise = drop_dominant_frequencies[-1] - drop_dominant_frequencies[-2]

    # Check that the rise in dominant frequencies is above the threshold
    if frequency_rise < rise_threshold:
        # print("Rise in dominant frequencies is below the threshold. Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
        return False

    # If all conditions are met, return True
    # print("Drop detected! Average dominant frequency: {:.3f}".format(safe_mean(dominant_frequencies)))
    return True


def calculate_dynamics(low_volumes, mid_volumes, high_volumes):
    # calculate mean volumes
    mean_low = safe_mean(low_volumes)
    mean_mid = safe_mean(mid_volumes)
    mean_high = safe_mean(high_volumes)

    # calculate standard deviation of volumes
    std_low = np.std(low_volumes)
    std_mid = np.std(mid_volumes)
    std_high = np.std(high_volumes)

    # calculate total mean and total standard deviation
    total_mean = 0 + mean_mid + mean_high
    total_std = 0 + std_mid + std_high

    # A higher total mean and a lower total standard deviation would indicate a more dynamic song
    dynamics = total_mean / total_std

    return dynamics


this_category = -1  # unknown


def categorize_song(raw_mean, low_volumes, mid_volumes, high_volumes, pitches):
    global this_category
    if raw_mean > 0.007:
        # Calculate mean volume in each frequency band
        low_mean_volume = safe_mean(low_volumes)
        mid_mean_volume = safe_mean(mid_volumes)
        high_mean_volume = safe_mean(high_volumes)

        # Calculate volume standard deviation in each frequency band
        low_std_volume = np.std(low_volumes)
        mid_std_volume = np.std(mid_volumes)
        high_std_volume = np.std(high_volumes)

        # Calculate maximum and minimum pitches
        this_pitches = pitches
        max_pitch = get_second_highest(this_pitches)
        min_pitch = np.min(this_pitches)

        # Print debug info
        """
        print(f"Low mean volume: {low_mean_volume}")
        print(f"Mid mean volume: {mid_mean_volume}")
        print(f"High mean volume: {high_mean_volume}")
        print(f"Low std volume: {low_std_volume}")
        print(f"Mid std volume: {mid_std_volume}")
        print(f"Max pitch: {max_pitch}")
        print(f"Min pitch: {min_pitch}")
        """
        # print(f"High std volume: {high_std_volume}")

        # If the mid volume is higher than high volume, and high volume standard deviation is relatively low, then classify as 'Melancholic'
        if mid_mean_volume > high_mean_volume and high_std_volume < 0.01 and min_pitch > 0 and max_pitch < 2000:
            this_category = 1  # relaxed music
            return this_category

        # If low volume is significantly higher than mid and high volumes and the standard deviation of the low volumes is high, then classify as 'Techno'
        elif (min_pitch == 0 and max_pitch > 4000) or (
                low_mean_volume > mid_mean_volume * 3 and low_mean_volume > high_mean_volume * 3 and low_std_volume > mid_std_volume):
            this_category = 2  # energetic music
            return this_category

        # If none of the above conditions are met, classify as 'Mixed'
        else:
            # this_category = 'Mixed'
            return this_category
    else:
        return 0  # no audio


def get_pitch(audiobuffer, p_detection):
    """
    This function gets the audio data from the provided stream and uses the
    provided pitch detection object to analyze the pitch of the audio.

    Args:
    stream : A PyAudio Stream object.
    pDetection : An Aubio pitch detection object.

    Returns:
    pitch : Detected pitch in Hz.
    """

    # Convert the audiobuffer to an array
    samples = np.fromstring(audiobuffer, dtype=aubio.float_type)
    # Use the pitch detection object to get the pitch
    pitch = p_detection(samples)[0]

    return pitch
