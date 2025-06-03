# MusicToLight3  Copyright (C) 2025  Felix Rau.
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

import pyaudio
import collections
import numpy as np
import aubio
from scipy.signal import ellip, sosfilt, sos2zpk, lfilter_zi
from helpers import *

class AudioAnalyzer:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        # Anzahl Samples für die rollenden Mittelwerte (wie average_samples)
        average_samples = int(sample_rate / buffer_size)

        # History-Collections wie bisher, aber jetzt als self.<name>
        self.volumes = collections.deque(maxlen=average_samples)
        self.heavyvols = collections.deque(maxlen=20)
        self.max_values = collections.deque(maxlen=20)
        self.heaviness_values = collections.deque(maxlen=average_samples)
        self.low_volumes = collections.deque(maxlen=average_samples)
        self.mid_volumes = collections.deque(maxlen=average_samples)
        self.high_volumes = collections.deque(maxlen=average_samples)
        self.last_counts = collections.deque(maxlen=5)
        self.previous_count_over = 0
        self.delta_values = collections.deque(maxlen=20)
        self.dominant_frequencies = collections.deque(maxlen=average_samples)
        self.heaviness_history = collections.deque(maxlen=average_samples)
        self.drop_history = collections.deque(maxlen=512)
        self.input_history = collections.deque(maxlen=average_samples)
        self.pitches = collections.deque(maxlen=average_samples)
        self.done_chase = collections.deque(maxlen=250)


analyzer = AudioAnalyzer(sample_rate=44100, buffer_size=1024)

def reset_analyzer_histories():
    analyzer.volumes.clear()
    analyzer.low_volumes.clear()
    analyzer.mid_volumes.clear()
    analyzer.high_volumes.clear()
    analyzer.heavyvols.clear()
    analyzer.max_values.clear()
    analyzer.delta_values.clear()
    analyzer.last_counts.clear()
    analyzer.heaviness_values.clear()
    analyzer.dominant_frequencies.clear()
    analyzer.heaviness_history.clear()
    analyzer.drop_history.clear()
    analyzer.input_history.clear()
    analyzer.pitches.clear()
    analyzer.done_chase.clear()

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
    Processes the audio signal and returns relevant metrics. This includes filter application,
    volume calculation, and heaviness analysis, updating several histories used for ongoing analysis.

    Returns:
        dict: A dictionary with the calculated audio metrics including volumes, signals, and states.
    """
    # Design the audio filters for different frequency bands
    thx_sos, thx_zi = design_filter(thx_band[0], thx_band[1])
    low_sos, low_zi = design_filter(low_band[0], low_band[1])
    mid_sos, mid_zi = design_filter(mid_band[0], mid_band[1])
    high_sos, high_zi = design_filter(high_band[0], high_band[1])

    heavy_counter = 0

    # Read and convert audio data from the input stream
    this_audio_buffer = line_in.read(int(buffer_size / 2), exception_on_overflow=False)
    signal_input = np.frombuffer(this_audio_buffer, dtype=np.float32)
    signal_max = np.max(signal_input)

    # Calculate the current volume and update the volume history
    current_volume = np.sqrt(np.mean(signal_input ** 2))
    analyzer.volumes.append(current_volume)

    # Calculate the mean volume from historical data
    mean_volume = safe_mean(analyzer.volumes)

    # Adjust the gain of the signal based on the mean volume
    signal, gain_factor = adjust_gain(mean_volume, signal_input)

    # Calculate relative volume
    relative_volume = 0 if mean_volume == 0 else current_volume / mean_volume
    relative_volume = min(1, max(0, relative_volume))

    # Filter the signal into different frequency bands and calculate their volumes
    low_signal, low_zi = sosfilt(low_sos, signal, zi=low_zi)
    analyzer.low_volumes.append(np.sqrt(np.mean(low_signal ** 2)))

    mid_signal, mid_zi = sosfilt(mid_sos, signal, zi=mid_zi)
    analyzer.mid_volumes.append(np.sqrt(np.mean(mid_signal ** 2)))

    high_signal, high_zi = sosfilt(high_sos, signal, zi=high_zi)
    analyzer.high_volumes.append(np.sqrt(np.mean(high_signal ** 2)))

    # Calculate means of the volumes for different frequency bands
    low_mean = np.mean(analyzer.low_volumes)
    mid_mean = np.mean(analyzer.mid_volumes)
    high_mean = np.mean(analyzer.high_volumes)

    # Calculate the relative volume for the low frequency band
    low_relative = np.mean(low_signal) / low_mean if low_mean != 0 else 0
    low_relative = min(1, max(0, low_relative))

    # Calculate total energy of the signal and relative energy
    energy = np.sum(signal ** 2)
    relative_energy = energy / len(signal)

    # Apply THX filter and analyze the output for heaviness
    thx_signal, thx_zi = sosfilt(thx_sos, signal, zi=thx_zi)
    thx_signal = thx_signal.astype(np.float32)
    analyzer.heavyvols.append(np.max(thx_signal))
    delta_value = np.max(thx_signal) - np.min(thx_signal)
    analyzer.max_values.append(np.max(thx_signal))
    analyzer.delta_values.append(delta_value)
    count_over = sum(1 for value in analyzer.max_values if value > 0.08)
    analyzer.last_counts.append(count_over)
    heavy, heavy_counter = is_heavy(signal, count_over, heavy_counter)
    heaviness = calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter)
    analyzer.heaviness_values.append(heaviness)

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
        "volumes": analyzer.volumes,
        "low_volumes": analyzer.low_volumes,
        "mid_volumes": analyzer.mid_volumes,
        "high_volumes": analyzer.high_volumes,
        "heavyvols": analyzer.heavyvols,
        "max_values": analyzer.max_values,
        "delta_values": analyzer.delta_values,
        "last_counts": analyzer.last_counts,
        "heaviness_values": analyzer.heaviness_values,
        "low_zi": low_zi,
        "mid_zi": mid_zi,
        "high_zi": high_zi,
        "thx_zi": thx_zi
    }


def adjust_gain(mean_volume, signal, target_volume=0.2, max_gain=100):
    """
    Adjusts the gain of the input signal based on the average volume to reach a target volume.

    Args:
    mean_volume : Average volume level.
    signal : Input signal array.
    target_volume : Target average volume level.
    max_gain : Maximum allowable gain factor.

    Returns:
    adjusted_signal : Signal after gain adjustment.
    gain_factor : Applied gain factor.
    """
    # Calculate the average volume level
    average_volume = mean_volume

    # Calculate the gain factor
    if average_volume <= 0:
        average_volume = 0.00001
    if average_volume < target_volume:
        gain_factor = target_volume / average_volume
    else:
        gain_factor = 1.0

    # Limit the gain factor to the maximum allowable value
    if gain_factor > max_gain:
        gain_factor = max_gain

    # Set values with an amplitude smaller than 0.01 to 0
    signal = signal * 1
    signal[np.abs(signal) < 0.014] = 0

    # Apply the gain factor to the signal
    adjusted_signal = signal * gain_factor

    return adjusted_signal, gain_factor


def design_filter(lowcut, highcut, ripple_db=0.5, stop_atten_db=40, order=3):
    """
    Designs a bandpass filter using an elliptical filter design to meet specified frequency characteristics.

    Args:
    lowcut : Low frequency cutoff for the filter.
    highcut : High frequency cutoff for the filter.
    ripple_db : Allowed ripple in the passband, in decibels.
    stop_atten_db : Minimum stopband attenuation, in decibels.
    order : Order of the filter.

    Globals:
    sample_rate : Sampling rate of the signal used for Nyquist frequency calculation.

    Returns:
    sos : Second-order sections representation of the filter.
    zi : Array of zeros for initial conditions.
    """
    global sample_rate
    # Calculate the Nyquist frequency, which is half the sampling rate.
    # This is used as the reference to normalize frequencies.
    nyquist = 0.5 * sample_rate

    # Normalize the low frequency cutoff against the Nyquist frequency.
    # This scales the frequency to a range between 0 and 1.
    low = lowcut / nyquist

    # Similarly, normalize the high frequency cutoff.
    high = highcut / nyquist

    # Create an elliptical bandpass filter (ellip) with the given specifications.
    # 'order' defines the complexity of the filter, 'ripple_db' the ripple in the passband,
    # 'stop_atten_db' the attenuation in the stopband, and '[low, high]' the frequency range.
    sos = ellip(order, ripple_db, stop_atten_db, [low, high], btype='band', output='sos')

    # Initialize an array of zeros for the initial conditions of the filter.
    # The shape of the array is based on the number of sections in the filter.
    zi = np.zeros((sos.shape[0], 2))

    return sos, zi


def is_heavy(this_signal, count_over, heavy_counter):
    """
    Determines if the signal is considered 'heavy' based on the amplitude of the signal,
    frequency of certain conditions, and historical data of changes.

    Args:
    this_signal : Input signal array.
    count_over : Count of occurrences where a certain condition is met.
    heavy_counter : Counter tracking occurrences deemed 'heavy'.

    Globals:
    delta_values : Array of delta values used to assess changes in signal characteristics.
    last_counts : Historical data of counts used for comparison.

    Returns:
    bool : True if the signal is considered 'heavy', otherwise False.
    int : Updated heavy counter.
    """
    # Check if historical counts data can be evaluated
    if len(analyzer.last_counts) == 5 and max(analyzer.last_counts) - min(analyzer.last_counts) >= 3:
        heavy_counter = 2  # Adjust the heavy counter based on historical data

    # Determine if the signal is 'heavy' by checking the maximum absolute signal value,
    # frequency of conditions being met, and a threshold from delta_values
    this_is_heavy = (np.max(np.abs(this_signal)) > 0.05 and
                     (count_over > 3 or heavy_counter > 0) and
                     np.max(analyzer.delta_values) > 0.3)
    return this_is_heavy, heavy_counter


def calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter):
    """
    Calculates a heaviness score between 0 and 10 based on the given parameters. The score
    is computed by normalizing each parameter to a range of 0 to 1, applying specific weights
    to each normalized parameter, and then scaling the result to a range of 0 to 10.

    Args:
    delta_value : Change in a certain signal characteristic.
    count_over : Count of occurrences where a certain threshold is exceeded.
    gain_factor : Gain factor applied to the signal.
    heavy_counter : Counter tracking occurrences deemed 'heavy'.

    Returns:
    float : Heaviness score between 0 and 10.
    """
    # Normalize the delta value to a maximum of 1 based on a predefined max change value.
    normalized_delta = min(delta_value / 0.5, 1)

    # Normalize the count over to a maximum of 1, assuming a max count of 10.
    normalized_count = min(count_over / 10.0, 1)

    # Normalize the gain factor to a maximum of 1, assuming a max gain of 5.
    normalized_gain = min(gain_factor / 5.0, 1)

    # Normalize the heavy counter to a maximum of 1, assuming a max value of 5.
    normalized_heavy_counter = min(heavy_counter / 5.0, 1)

    # Calculate the overall heaviness score using weighted contributions from each parameter
    heaviness = (0.4 * normalized_delta + 0.3 * normalized_count +
                 0.2 * normalized_gain + 0.1 * normalized_heavy_counter)

    # Scale the computed heaviness to the desired range of 0 to 10 for final scoring
    heaviness *= 10

    return heaviness


def dominant_frequency(this_signal):
    """
    Calculates the dominant frequency in a signal using a fast Fourier transform (FFT).

    Args:
    this_signal : Numpy array of signal values.

    Globals:
    sample_rate : Sampling rate of the signal, in Hz.

    Returns:
    float : Dominant frequency in Hertz (Hz).
    """
    global sample_rate
    # Check if the signal is effectively zero to avoid unnecessary calculations
    if np.all(np.isclose(this_signal, 0, atol=0.01)):
        return 0  # Return 0 Hz if the signal is close to zero

    # Perform FFT on the signal after removing the mean to center it
    fft_result = np.fft.rfft(this_signal - np.mean(this_signal))
    # Find the index of the maximum in the FFT result, which corresponds to the dominant frequency
    dominant_frequency_index = np.argmax(np.abs(fft_result))
    # Convert the index to actual frequency using the formula
    return dominant_frequency_index * sample_rate / len(this_signal)


def detect_drop(volume_mean, heavy, drop_length=40, rise_threshold=15, volume_rise_threshold=0.08,
                dominant_frequencies_threshold=300, heavy_threshold=0.8, drop_threshold=1, heavy_return_threshold=2):
    """
    Detects if a 'drop' event has occurred in an audio signal based on various thresholds and historical data.

    Args:
    volume_mean : Current average volume of the signal.
    heavy : Boolean indicating if the current state is considered 'heavy'.
    drop_length : Number of recent data points to consider for drop detection.
    rise_threshold : Threshold for considering a frequency rise significant.
    volume_rise_threshold : Volume level at which to increase the rise threshold.
    dominant_frequencies_threshold : Minimum average frequency required to negate a drop.
    heavy_threshold : Proportion of 'heavy' states required before a drop.
    drop_threshold : Minimum number of drops required to consider detecting another.
    heavy_return_threshold : Number of consecutive 'heavy' states to consider the end of a drop.

    Globals:
    dominant_frequencies : List of dominant frequencies in recent data points.
    heaviness_history : List of boolean indicating heavy states in recent data points.
    drop_history : List of detected drops.

    Returns:
    bool : True if a drop is detected, otherwise False.
    """

    # Check if there is sufficient data to analyze for a drop
    if len(analyzer.dominant_frequencies) < drop_length or len(analyzer.heaviness_history) < drop_length:
        return False

    # Adjust the rise threshold based on the current volume mean
    effective_rise_threshold = rise_threshold * 1.5 if volume_mean > volume_rise_threshold else rise_threshold

    # Check for enough drop events and that the current state is not 'heavy'
    total_drops = sum(analyzer.drop_history)
    if total_drops >= drop_threshold and not heavy:
        return True

    # Get the last relevant values for frequencies and heaviness
    drop_dominant_frequencies = np.array(analyzer.dominant_frequencies)[-drop_length:]
    heavy_history = np.array(analyzer.heaviness_history)[-drop_length:]

    # Check for the end of a drop condition: two consecutive 'heavy' states
    if heavy_history[-heavy_return_threshold:].all():
        return False

    # Check for the start of a drop: transition from heavy to non-heavy
    if heavy or not heavy_history[:-heavy_return_threshold].any():
        return False

    # Ensure there were enough 'heavy' states before the drop
    if heavy_history.sum() < heavy_threshold * drop_length:
        return False

    # Verify that the average dominant frequency during the drop is above the threshold
    if drop_dominant_frequencies.mean() <= dominant_frequencies_threshold:
        return False

    # Calculate the frequency rise between the last two data points
    frequency_rise = drop_dominant_frequencies[-1] - drop_dominant_frequencies[-2]

    # Check if the frequency rise is above the effective threshold
    if frequency_rise < effective_rise_threshold:
        return False

    # If all conditions are met, a drop is detected
    return True


def calculate_band_intensities(signal, sample_rate=44100, buffer_size=128):
    signal = signal[:buffer_size]  # Nur die ersten buffer_size Samples verwenden
    fft_result = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)

    # Definiere 12 Frequenzbänder (Beispielhaft)
    bands = [0, 300, 600, 1000, 1500, 2000, 3000, 4000, 5000, 8000, 12000, 16000, 20000]
    band_indices = [np.where((freqs >= bands[i]) & (freqs < bands[i + 1]))[0] for i in range(len(bands) - 1)]

    band_intensities = [np.sqrt(np.mean(fft_magnitude[indices] ** 2)) for indices in band_indices]
    return band_intensities


class BandScaler:
    def __init__(self, num_bands=12, base_history_length=400, min_scale=0.02, max_scale=2,
                 fast_adjust_threshold=0.5, slow_adjust_threshold=0.2):
        self.num_bands = num_bands
        self.base_history_length = base_history_length  # Grundwert für die History-Länge
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fast_adjust_threshold = fast_adjust_threshold  # Ab dieser Lautstärke passt sich die History schneller an
        self.slow_adjust_threshold = slow_adjust_threshold  # Unter dieser Lautstärke passt sich die History langsamer an

        # Initialisiere eine Liste von Deques, um die historischen Daten für jedes Band zu speichern
        self.band_histories = [deque(maxlen=base_history_length) for _ in range(num_bands)]

    def update_and_scale(self, band_intensities):
        scaled_intensities = []

        for i in range(self.num_bands):
            # Füge die aktuelle Intensität zur Historie des Bands hinzu
            current_intensity = band_intensities[i]

            # Dynamisch die History-Länge anpassen, basierend auf der Lautstärke
            if current_intensity > self.fast_adjust_threshold:
                # Laute Töne -> kürzere History
                new_history_length = int(self.base_history_length / 2)  # Halbierung der History-Länge bei lauten Tönen
            elif current_intensity < self.slow_adjust_threshold:
                # Leise Töne -> längere History
                new_history_length = int(
                    self.base_history_length * 1.5)  # Verlängerung der History-Länge bei leisen Tönen
            else:
                # Normalfall -> Basis-History-Länge
                new_history_length = self.base_history_length

            # Stelle sicher, dass die Deque die angepasste Länge hat
            self.band_histories[i] = deque(self.band_histories[i], maxlen=new_history_length)
            self.band_histories[i].append(current_intensity)

            # Berechne den Durchschnitt der Historie für das Band
            avg_intensity = np.mean(self.band_histories[i])

            # Berechne den Skalierungsfaktor für das Band
            scale_factor = np.clip(1 / (avg_intensity + 1e-6), self.min_scale, self.max_scale)

            # Skaliere die aktuelle Intensität basierend auf dem Skalierungsfaktor
            scaled_intensity = current_intensity * scale_factor
            scaled_intensities.append(scaled_intensity)

        return scaled_intensities


band_scaler = BandScaler()


def process_audio_and_scale(signal):
    # Berechne die Bandintensitäten
    band_intensities = calculate_band_intensities(signal)

    # Skaliere die Bandintensitäten dynamisch
    scaled_intensities = band_scaler.update_and_scale(band_intensities)

    return scaled_intensities
