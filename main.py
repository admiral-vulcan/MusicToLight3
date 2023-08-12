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

import pyaudio
import collections
import time
import math
import numpy as np
import aubio
from scipy.signal import ellip, sosfilt, sos2zpk, lfilter_zi
from aud_proc import *
from eurolite_t36 import *
from scanner import *
from led_strip import *
from hdmi import *

# Filter-Einstellungen
thx_band = (1, 120)
low_band = (1, 300)
mid_band = (300, 2000)
high_band = (2000, 16000)

# PyAudio Konfiguration
buffer_size = 1024
hop_size = buffer_size // 2
pyaudio_format = pyaudio.paFloat32
n_channels = 1
sample_rate = 44100
device_index = 0

# Create a new pitch detection object
pDetection = aubio.pitch("default", buffer_size, hop_size, sample_rate)
pDetection.set_unit("Hz")  # We want the result as frequency in Hz
pDetection.set_tolerance(0.8)

# Erstellen Sie Aubio-Beat-Detection-Objekt
aubio_onset = aubio.onset("complex", buffer_size, hop_size, sample_rate)

# Design the elliptic filter
thx_sos, thx_zi = design_filter(thx_band[0], thx_band[1], sample_rate)
low_sos, low_zi = design_filter(low_band[0], low_band[1], sample_rate)
mid_sos, mid_zi = design_filter(mid_band[0], mid_band[1], sample_rate)
high_sos, high_zi = design_filter(high_band[0], high_band[1], sample_rate)
# PyAudio Objekt
p = pyaudio.PyAudio()

# Geräte-Index basierend auf dem Gerätenamen ermitteln
desired_device_index = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if 'ICUSBAUDIO7D: USB Audio' in info["name"]:
        desired_device_index = i
        break

# Das gewünschte Gerät in Ihrem `open`-Aufruf verwenden
if desired_device_index is not None:
    line_in = p.open(format=pyaudio_format,
                     channels=n_channels,
                     rate=sample_rate,
                     input=True,
                     input_device_index=desired_device_index,
                     frames_per_buffer=buffer_size)
else:
    print("Kein passendes Audiogerät gefunden!")
    exit()

# Setzen Sie die Anzahl der Proben für die Durchschnittsberechnung
average_samples = int(5 * sample_rate / buffer_size)  # average over ~5 seconds
average_heavy_samples = int(sample_rate / buffer_size)  # average over ~1 second

# Initialisieren Sie eine deque (double-ended queue) mit einer festen Länge
volumes = collections.deque(maxlen=average_samples)
heavyvols = collections.deque(maxlen=20)
max_values = collections.deque(maxlen=20)  # store the maximum values for ~2 seconds
heaviness_values = collections.deque(maxlen=average_samples)  # Sammeln der Heaviness-Werte

low_volumes = collections.deque(maxlen=average_samples)
mid_volumes = collections.deque(maxlen=average_samples)
high_volumes = collections.deque(maxlen=average_samples)

# Initialisieren Sie last_counts, previous_count_over und heavy_counter vor der Schleife
last_counts = collections.deque(maxlen=5)
previous_count_over = 0
heavy_counter = 0
delta_values = collections.deque(maxlen=20)

dominant_frequencies = collections.deque(maxlen=average_samples)
heaviness_history = collections.deque(maxlen=average_samples)
drop_history = collections.deque(maxlen=512)
input_history = collections.deque(maxlen=average_samples)
pitches = collections.deque(maxlen=average_samples)

done_chase = deque(maxlen=int(250))  # adjust as needed

runtime_bit = 0
runtime_byte = 0
runtime_kb = 0
runtime_mb = 0

previous_heavy = True

print("")
print("        MusicToLight3  Copyright (C) 2023  Felix Rau")
print("        This program is licensed under the terms of the GNU General Public License version 3.")
print("        It comes with ABSOLUTELY NO WARRANTY; for details see README.md.")
print("        This is free software, and you are welcome to redistribute it")
print("        under certain conditions; see LICENSE.md.")
print("")
print("        Initialising devices.")
print("")

# initialise devices
init_hdmi()
hdmi_draw_centered_text(
    "MusicToLight3  Copyright (C) 2023  Felix Rau\n\n\nThis program is licensed under the terms of the \nGNU General Public License version 3.\nIt is open source, free, and comes with ABSOLUTELY NO WARRANTY.\n\n\nInitialising devices...")
scan_reset(1)
scan_reset(2)
hdmi_intro_animation()
# scan_in_thread(scan_reset, (1,))
# scan_in_thread(scan_reset, (2,))
# set_eurolite_t36(5, 0, 0, 0, 0, 0)
# set_eurolite_t36(5, 0, 0, 0, 255, 0)
color_wipe(Color(0, 0, 0), 0)
print("        Listening... Press Ctrl+C to stop.")
print("")
hdmi_draw_black()

try:
    while True:
        runtime_bit += 1

        if runtime_bit > 255:
            runtime_bit = 0
            runtime_byte += 1

        if runtime_byte > 1024:
            runtime_byte = 0
            runtime_kb += 1

        if runtime_kb > 1024:
            runtime_kb = 0
            runtime_mb += 1

        # print(runtime_mb, runtime_kb, runtime_byte, runtime_bit)

        audiobuffer = line_in.read(int(buffer_size / 2), exception_on_overflow=False)
        signal_input = np.frombuffer(audiobuffer, dtype=np.float32)

        # Adjust gain
        signal, gain_factor = adjust_gain(volumes, signal_input)

        # Calculate the volume of the current signal
        volume = np.sqrt(safe_mean(signal ** 2))
        # Add the current volume to the list of previous volumes
        volumes.append(volume)

        # apply low frequency filter to signal
        low_signal, low_zi = sosfilt(low_sos, signal, zi=low_zi)
        low_volume = np.sqrt(safe_mean(low_signal ** 2))
        low_volumes.append(low_volume)

        # apply mid frequency filter to signal
        mid_signal, mid_zi = sosfilt(mid_sos, signal, zi=mid_zi)
        mid_volume = np.sqrt(safe_mean(mid_signal ** 2))
        mid_volumes.append(mid_volume)

        # apply high frequency filter to signal
        high_signal, high_zi = sosfilt(high_sos, signal, zi=high_zi)
        high_volume = np.sqrt(safe_mean(high_signal ** 2))
        high_volumes.append(high_volume)

        low_mean = compute_mean_volume(low_volumes)
        mid_mean = compute_mean_volume(mid_volumes)
        high_mean = compute_mean_volume(high_volumes)

        hdmi_matrix = generate_matrix(low_signal, mid_signal, high_signal, low_mean, mid_mean, high_mean)
        transposed_hdmi_matrix = list(map(list, zip(*hdmi_matrix)))

        # Calculate energies
        energy = np.sum(signal ** 2)
        db_energy = 10 * np.log10(energy)
        relative_energy = energy / len(signal)
        db_relative_energy = 10 * np.log10(relative_energy)

        thx_signal, zi = sosfilt(thx_sos, signal, zi=thx_zi)  # Apply the filter with the previous final state
        thx_signal = thx_signal.astype(np.float32)

        heavyvols.append(np.max(thx_signal))
        heavyvol = safe_mean(heavyvols)
        max_value = np.max(thx_signal)
        min_value = np.min(thx_signal)
        delta_value = max_value - min_value
        max_values.append(max_value)
        delta_values.append(delta_value)
        count_over = sum(1 for value in max_values if value > 0.08)
        last_counts.append(count_over)

        # Check if signal is heavy
        heavy, heavy_counter = is_heavy(signal, delta_values, count_over, max_values, last_counts, heavy_counter)
        heaviness = calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter)
        heaviness_values.append(heaviness)  # Sammeln der aktuellen Heaviness-Werte

        dominant_freq = dominant_frequency(signal, sample_rate)
        dominant_frequencies.append(dominant_freq)
        heaviness_history.append(heavy)

        drop = detect_drop(safe_mean(volumes), heavy, dominant_frequencies, heaviness_history, drop_history)
        drop_history.append(drop)

        previous_heavy = heavy

        # Beat erkennen
        is_beat = aubio_onset(thx_signal)

        pitch = get_pitch(audiobuffer, pDetection)
        pitches.append(pitch)
        # print("Detected pitch: %.2f Hz" % pitch)
        # dynamic = calculate_dynamics(low_volumes, mid_volumes, high_volumes)
        # print(dynamic)
        # category = categorize_song(np.max(signal_input), low_volumes, mid_volumes, high_volumes, pitches)
        # print(category)

        # Speichern Sie count_over für die nächste Iteration
        previous_count_over = count_over

        if heavy and 1 in list(done_chase)[-10:]:
            # strobe here
            kill_current_hdmi()
            # make all dark!
            scan_closed(1)
            scan_closed(2)
            hdmi_draw_black()
            led_strobe_effect(10, 75)
            hdmi_intro_animation()
            done_chase.clear()

        # hdmi numbers here
        hdmi_draw_matrix(transposed_hdmi_matrix)

        red = int(energy * 10)
        if red > 255:
            red = 255

        y = ((int(energy * 10) - 60) * 1.75)
        if y < 0:
            y = 0
        if y > 255:
            y = 255

        # print(y)

        # scanner operates here
        x = int(exponential_decrease(red))

        # print(x)

        # DMX lamps and LED strip operate here
        done_chase.append(0)
        scan_gobo(1, 7, 17)
        scan_gobo(2, 7, 17)
        scan_in_thread(scan_color, (1, "purple"))
        scan_in_thread(scan_color, (2, "blue"))

        if heavy:
            scan_opened(1)
            scan_opened(2)
            scan_in_thread(scan_axis, (1, y, x))  # der vordere
            scan_in_thread(scan_axis, (2, x, y))  # der hintere
            led_music_visualizer(np.max(signal_input))
            # color_flow(runtime_bit, np.max(signal_input))
            drop = False
            # Überprüfen, ob ein Beat erkannt wurde
            if is_beat:
                drop_history.clear()
                # print("**************************************************************")
                # print("*************************** Beat! ****************************")
                # print("**************************************************************")
                set_eurolite_t36(5, 255, 0, 50, 255, 0)
            else:
                # print("**************************************************************")
                # print("*************************** Heavy! ***************************")
                # print("**************************************************************")
                set_eurolite_t36(5, red, 0, 255, 255, 0)

            if heavy_counter > 0:
                heavy_counter -= 1  # Reduzieren Sie heavy_counter um 1
        else:
            # scan_closed(1)
            # scan_closed(2)
            scan_go_home(1)
            scan_go_home(2)
            # scan_in_thread(scan_axis, (1, 255, 255))
            # scan_in_thread(scan_axis, (2, 255, 255))
            if np.max(signal_input) > 0.007:
                input_history.append(1.0)
                if not heavy and not drop:
                    color_flow(runtime_bit, np.max(signal_input))
                    # print("** ? **", np.max(signal_input))

                if 0 < sum(drop_history) < 16 and drop:
                    # color_wipe(Color(0, 0, 0), 0)
                    # print("**************************************************************")
                    # print("*********************** Drop detected. ***********************")
                    # print("**************************************************************")
                    color_flow(runtime_bit, np.max(signal_input))
                    set_eurolite_t36(5, invert(sum(drop_history) - 128, 256), 0, 100, 255, 0)

                if 16 <= sum(drop_history) < 32 and drop:
                    # color_wipe(Color(0, 0, 0), 0)
                    # print("**************************************************************")
                    # print("*********************** Drop rises... ************************")
                    # print("**************************************************************")
                    color_flow(runtime_bit, np.max(signal_input))
                    set_eurolite_t36(5, 0, 0, invert(sum(drop_history), 256), 255, 0)

                if sum(drop_history) >= 32 and drop:  # not if too often!
                    heaviness_history.clear()
                    if 1 not in done_chase:
                        hdmi_outro_animation()
                        scan_closed(1)
                        scan_closed(2)
                        set_eurolite_t36(5, 0, 0, 0, 255, 0)
                        theater_chase(Color(127, 127, 127), 52)
                        hdmi_intro_animation()
                        scan_opened(1)
                        scan_opened(2)
                    done_chase.append(1)
                    # print("**************************************************************")
                    # print("********************** Drop persistent! **********************")
                    # print("**************************************************************")
            else:
                set_eurolite_t36(5, 0, 0, 0, 255, 0)
                input_history.append(0.0)
                if safe_mean(input_history) < 0.5:
                    drop = False
                    heavy = False
                    low_volumes.clear()
                    mid_volumes.clear()
                    high_volumes.clear()
                    pitches.clear()
                    drop_history.clear()
                    heaviness_history.clear()
                    color_flow(runtime_bit, np.max(signal_input), 20)  # last arg is brightness divider
                    # print("** no input **")

except KeyboardInterrupt:
    hdmi_outro_animation()
    print("")
    print("\n        Ending program...")
    color_wipe(Color(0, 0, 0), 0)
    # scan_reset(1)
    # scan_reset(2)
    scan_closed(1)
    scan_closed(2)
    set_eurolite_t36(5, 0, 0, 0, 0, 0)
    line_in.close()
    p.terminate()

    if current_hdmi_thread and current_hdmi_thread.is_alive():
        current_hdmi_thread.join()

    time.sleep(2)
    hdmi_draw_centered_text(
        "MusicToLight3  Copyright (C) 2023  Felix Rau\n\n\nThis program is licensed under the terms of the \nGNU General Public License version 3.\nIt is open source, free, and comes with ABSOLUTELY NO WARRANTY.\n\n\nProgram ended gracefully.")
    time.sleep(5)

    print("")
    print("\n        Program ended gracefully.")
    print("")
    print("        MusicToLight3  Copyright (C) 2023  Felix Rau")
    print("        This program is licensed under the terms of the GNU General Public License version 3.")
    print("        It comes with ABSOLUTELY NO WARRANTY; for details see README.md.")
    print("        This is free software, and you are welcome to redistribute it")
    print("        under certain conditions; see LICENSE.md.")
    print("")
