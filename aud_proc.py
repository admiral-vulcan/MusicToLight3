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


import os
import pyaudio
import collections
import math
import numpy as np
import aubio
from scipy.signal import ellip, sosfilt, sos2zpk, lfilter_zi
from helpers import *
from pyloudnorm import Meter
from scipy.fft import fft
import sys

os.environ['ALSA_LOG_LEVEL'] = 'error'

# Filter configurations
thx_band = (1, 80)  # 120
low_band = (1, 300)  # 300
mid_band = (300, 2000)
high_band = (2000, 16000)

# PyAudio settings
buffer_size = 1024
hop_size = buffer_size // 2
pyaudio_format = pyaudio.paFloat32
n_channels = 1
sample_rate = 44100
device_index = 0

# Define the number of samples for average calculations
average_samples = int(sample_rate / buffer_size)  # why??

# Initialize data collections
volumes = collections.deque(maxlen=average_samples)
heavyvols = collections.deque(maxlen=20)
max_values = collections.deque(maxlen=20)
heaviness_values = collections.deque(maxlen=average_samples)

low_volumes = collections.deque(maxlen=average_samples)
mid_volumes = collections.deque(maxlen=average_samples)
high_volumes = collections.deque(maxlen=average_samples)

last_counts = collections.deque(maxlen=5)
previous_count_over = 0
delta_values = collections.deque(maxlen=20)

dominant_frequencies = collections.deque(maxlen=average_samples)
heaviness_history = collections.deque(maxlen=average_samples)
drop_history = collections.deque(maxlen=512)
input_history = collections.deque(maxlen=average_samples)
pitches = collections.deque(maxlen=average_samples)

done_chase = collections.deque(maxlen=int(250))


def get_second_highest(values):
    """
    Returns the second highest value in the given array using a partition-based selection algorithm.
    This is more efficient than sorting for large arrays.

    Args:
    values : A NumPy array.

    Returns:
    second_highest : Second highest value in the array.
    """
    if len(values) > 1:
        return np.partition(values, -2)[-2]
    return values[0] if values.size else None


def adjust_gain(mean_volume, signal, target_volume=0.2, max_gain=100):
    """
    Adjusts the gain of the input signal based on the average volume to reach a target volume.

    Args:
    this_volumes : Array of volume levels.
    signal : Input signal array.
    target_volume : Target average volume level.
    max_gain : Maximum allowable gain factor.

    Returns:
    adjusted_signal : Signal after gain adjustment.
    gain_factor : Applied gain factor.
    """
    # Berechne den durchschnittlichen Lautstärkepegel
    # average_volume = safe_mean(this_volumes)
    average_volume = mean_volume

    # Berechne den Verstärkungsfaktor
    if average_volume <= 0:
        average_volume = 0.00001
    if average_volume < target_volume:
        gain_factor = target_volume / average_volume
    else:
        gain_factor = 1.0

    # Begrenze den Verstärkungsfaktor auf den maximal zulässigen Wert
    if gain_factor > max_gain:
        gain_factor = max_gain

    # Setze Werte mit einem Betrag kleiner als 0,01 auf 0
    signal = signal * 1
    signal[np.abs(signal) < 0.014] = 0

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
    # bug: heavy_counter always zero
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
    """
    Calculate the dominant frequency in a signal using fast Fourier transform.

    Args:
    signal : Numpy array of signal values.
    sample_rate : Sampling rate of the signal.

    Returns:
    dominant_frequency_hz : Dominant frequency in Hertz.
    """
    if np.all(np.isclose(signal, 0, atol=0.01)):
        return 0

    fft_result = np.fft.rfft(signal - np.mean(signal))
    dominant_frequency_index = np.argmax(np.abs(fft_result))
    return dominant_frequency_index * sample_rate / len(signal)


def detect_drop(volume_mean, heavy, dominant_frequencies, heaviness_history, drop_history, drop_length=40,
                rise_threshold=15, volume_rise_threshold=0.08, dominant_frequencies_threshold=300, heavy_threshold=0.8,
                drop_threshold=1, heavy_return_threshold=2):
    # Überprüfung auf ausreichende Datenlänge
    if len(dominant_frequencies) < drop_length or len(heaviness_history) < drop_length:
        return False

    # Anpassen des rise_threshold basierend auf volume_mean
    effective_rise_threshold = rise_threshold * 1.5 if volume_mean > volume_rise_threshold else rise_threshold

    # Überprüfung, ob genügend Drops aufgetreten sind und ob nicht aktuell ein "heavy" Zustand ist
    total_drops = sum(drop_history)
    if total_drops >= drop_threshold and not heavy:
        return True

    # Letzte relevante Werte für Frequenzen und Heaviness
    drop_dominant_frequencies = np.array(dominant_frequencies)[-drop_length:]
    heavy_history = np.array(heaviness_history)[-drop_length:]

    # Überprüfung des Endes des Drops: heavy ist True zweimal hintereinander
    if heavy_history[-heavy_return_threshold:].all():
        return False

    # Überprüfung des Starts des Drops: Übergang von heavy zu nicht-heavy
    if heavy or not heavy_history[:-heavy_return_threshold].any():
        return False

    # Sicherstellen, dass vor dem Drop genügend 'heavy' Zustände vorhanden waren
    if heavy_history.sum() < heavy_threshold * drop_length:
        return False

    # Überprüfung, ob die durchschnittliche dominante Frequenz während des Drops oberhalb des Schwellenwertes liegt
    if drop_dominant_frequencies.mean() <= dominant_frequencies_threshold:
        return False

    # Berechnung des Anstiegs der dominanten Frequenzen
    frequency_rise = drop_dominant_frequencies[-1] - drop_dominant_frequencies[-2]

    # Überprüfung, ob der Frequenzanstieg über dem effektiven Schwellenwert liegt
    if frequency_rise < effective_rise_threshold:
        return False

    # Wenn alle Bedingungen erfüllt sind
    return True


# Initialize Aubio beat detection
# Methods: <default|energy|hfc|complex|phase|specdiff|kl|mkl|specflux>
aubio_onset = aubio.onset("default", buffer_size, hop_size, sample_rate)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Retrieve device index based on device name
desired_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if 'ICUSBAUDIO7D: USB Audio' in info["name"]:
        desired_device_index = i
        break

# Use the desired device to open audio stream
if desired_device_index is not None:
    line_in = p.open(format=pyaudio_format,
                     channels=n_channels,
                     rate=sample_rate,
                     input=True,
                     input_device_index=desired_device_index,
                     frames_per_buffer=buffer_size)
else:
    print("No suitable audio device found!")
    exit()


def process_audio_buffer():
    """
    Processes the audio signal and returns relevant metrics.

    Args:
        line_in: PyAudio stream object.
        buffer_size: Size of the buffer for the audio stream.
        low_sos, mid_sos, high_sos, thx_sos: SOS filter coefficients.
        volumes, low_volumes, mid_volumes, high_volumes, heavyvols,
        max_values, delta_values, last_counts, heaviness_values: Histories for various metrics.
        low_zi, mid_zi, high_zi, thx_zi: Initial states for the filters.

        Returns:
            dict: A dictionary with the calculated audio metrics.
    """
    # Design the audio filters
    thx_sos, thx_zi = design_filter(thx_band[0], thx_band[1], sample_rate)
    low_sos, low_zi = design_filter(low_band[0], low_band[1], sample_rate)
    mid_sos, mid_zi = design_filter(mid_band[0], mid_band[1], sample_rate)
    high_sos, high_zi = design_filter(high_band[0], high_band[1], sample_rate)

    heavy_counter = 0

    this_audio_buffer = line_in.read(int(buffer_size / 2), exception_on_overflow=False)
    signal_input = np.frombuffer(this_audio_buffer, dtype=np.float32)
    signal_max = np.max(signal_input)

    current_volume = np.sqrt(np.mean(signal_input ** 2))
    volumes.append(current_volume)

    mean_volume = safe_mean(volumes)

    signal, gain_factor = adjust_gain(mean_volume, signal_input)

    relative_volume = 0 if mean_volume == 0 else current_volume / mean_volume
    relative_volume = min(1, max(0, relative_volume))

    low_signal, low_zi = sosfilt(low_sos, signal, zi=low_zi)
    low_volumes.append(np.sqrt(np.mean(low_signal ** 2)))

    mid_signal, mid_zi = sosfilt(mid_sos, signal, zi=mid_zi)
    mid_volumes.append(np.sqrt(np.mean(mid_signal ** 2)))

    high_signal, high_zi = sosfilt(high_sos, signal, zi=high_zi)
    high_volumes.append(np.sqrt(np.mean(high_signal ** 2)))

    low_mean = np.mean(low_volumes)
    mid_mean = np.mean(mid_volumes)
    high_mean = np.mean(high_volumes)

    low_relative = np.mean(low_signal) / low_mean if low_mean != 0 else 0
    low_relative = min(1, max(0, low_relative))

    energy = np.sum(signal ** 2)
    relative_energy = energy / len(signal)

    thx_signal, thx_zi = sosfilt(thx_sos, signal, zi=thx_zi)
    thx_signal = thx_signal.astype(np.float32)
    heavyvols.append(np.max(thx_signal))
    delta_value = np.max(thx_signal) - np.min(thx_signal)
    max_values.append(np.max(thx_signal))
    delta_values.append(delta_value)
    count_over = sum(1 for value in max_values if value > 0.08)
    last_counts.append(count_over)
    heavy, heavy_counter = is_heavy(signal, delta_values, count_over, max_values, last_counts, heavy_counter)
    heaviness = calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter)
    heaviness_values.append(heaviness)

    return {
        "signal": signal,
        "thx_signal": thx_signal,
        "low_signal": low_signal,
        "mid_signal": mid_signal,
        "high_signal": high_signal,
        "low_mean": low_mean,
        "mid_mean": mid_mean,
        "high_mean": high_mean,
        "signal_max": signal_max,
        "relative_volume": relative_volume,
        "mean_volume": mean_volume,
        "low_relative": low_relative,
        "energy": energy,
        "relative_energy": relative_energy,
        "heavy": heavy,
        "heaviness": heaviness,
        "heavy_counter": heavy_counter,
        # Zustände zurückgeben, falls nötig
        "volumes": volumes,
        "low_volumes": low_volumes,
        "mid_volumes": mid_volumes,
        "high_volumes": high_volumes,
        "heavyvols": heavyvols,
        "max_values": max_values,
        "delta_values": delta_values,
        "last_counts": last_counts,
        "heaviness_values": heaviness_values,
        # Filterzustände zurückgeben
        "low_zi": low_zi,
        "mid_zi": mid_zi,
        "high_zi": high_zi,
        "thx_zi": thx_zi
    }
