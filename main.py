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
from pyloudnorm import Meter
from scipy.fft import fft
import aubio

import RPi.GPIO as GPIO
import redis
import json
from scipy.signal import ellip, sosfilt, sos2zpk, lfilter_zi
from aud_proc import *
from eurolite_t36 import *
from scanner import *
from laser_show import *
from led_strip import *
from hdmi import *
from smoker import *
import os
import sys
from com_udp import *
import argparse

# from vid import *

UDP_LED_COUNT = 45
UDP_IP_ADDRESS = "192.168.1.152"  # musictolight-led1
UDP_PORT = 4210

parser = argparse.ArgumentParser(description='MusicToLight3')
parser.add_argument('--fastboot', action='store_true', help='Activates Fastboot-Mode. Deactivates calibrating.')
args = parser.parse_args()

# Setting up communication with web server via Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_client.set('strobe_mode', 'auto')
redis_client.set('smoke_mode', 'auto')
redis_client.set('panic_mode', 'off')
redis_client.set('play_videos_mode', 'auto')

# Set master colors (TODO should later be changeable via web interface)
st_color_name = "blue"
nd_color_name = "red"
st_prim_color = get_rgb_from_color_name(st_color_name)
nd_prim_color = get_rgb_from_color_name(nd_color_name)

# Translate to RGB-Bytes
st_r, st_g, st_b = st_prim_color
nd_r, nd_g, nd_b = nd_prim_color

# Calculate the average of the two colors
average_color = tuple((a + b) // 2 for a, b in zip(st_prim_color, nd_prim_color))

# Find the highest value in the average
max_val = max(average_color)

# Calculate the factor to scale the highest value to 255
if max_val != 0:  # Prevent division by zero
    factor = 255 / max_val
else:
    factor = 0

# Multiply each color value in the average by the factor
secondary_color = tuple(int(val * factor) for val in average_color)

redis_client.set('st_prim_color', json.dumps(st_prim_color))
redis_client.set('nd_prim_color', json.dumps(nd_prim_color))
redis_client.set('secondary_color', json.dumps(secondary_color))

# init smoke
init_smoke()


def redis_get_colors():
    """
    Retrieve the color values from Redis and assign them to the global variables.
    """
    global st_prim_color
    global nd_prim_color
    global secondary_color
    global st_r, st_g, st_b
    global nd_r, nd_g, nd_b

    st_prim_color = tuple(json.loads(redis_client.get('st_prim_color')))
    nd_prim_color = tuple(json.loads(redis_client.get('nd_prim_color')))
    secondary_color = tuple(json.loads(redis_client.get('secondary_color')))

    # Translate to RGB-Bytes
    st_r, st_g, st_b = st_prim_color
    nd_r, nd_g, nd_b = nd_prim_color


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

# Initialize LUFS-loudness recognition
meter = Meter(sample_rate)

# Initialize pitch detection
pDetection = aubio.pitch("default", buffer_size, hop_size, sample_rate)
pDetection.set_unit("Hz")
pDetection.set_tolerance(0.8)

# Initialize Aubio beat detection
aubio_onset = aubio.onset("complex", buffer_size, hop_size, sample_rate)

# Design the audio filters
thx_sos, thx_zi = design_filter(thx_band[0], thx_band[1], sample_rate)
low_sos, low_zi = design_filter(low_band[0], low_band[1], sample_rate)
mid_sos, mid_zi = design_filter(mid_band[0], mid_band[1], sample_rate)
high_sos, high_zi = design_filter(high_band[0], high_band[1], sample_rate)

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

# Define the number of samples for average calculations
average_samples = int(5 * sample_rate / buffer_size)
average_heavy_samples = int(sample_rate / buffer_size)

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
heavy_counter = 0
delta_values = collections.deque(maxlen=20)

dominant_frequencies = collections.deque(maxlen=average_samples)
heaviness_history = collections.deque(maxlen=average_samples)
drop_history = collections.deque(maxlen=512)
input_history = collections.deque(maxlen=average_samples)
pitches = collections.deque(maxlen=average_samples)

done_chase = deque(maxlen=int(250))

runtime_bit = 0
runtime_byte = 0
runtime_kb = 0
runtime_mb = 0

no_drop_count = 0

previous_heavy = True

print("")
print("\nProgram ended gracefully.\n")
print("MusicToLight3  Copyright (C) 2023  Felix Rau")
print("This program is licensed under the terms of the ")
print("GNU General Public License version 3.")
print("It comes with ABSOLUTELY NO WARRANTY; for details see README.md.")
print("This is free software, and you are welcome to redistribute it")
print("under certain conditions; see LICENSE.md.\n")
print("")
print("        Initializing devices.")
print("")

# initialise devices
init_hdmi()
hdmi_draw_centered_text(
    "MusicToLight3  Copyright (C) 2023  Felix Rau\n\n\n"
    "This program is licensed under the terms of the \n"
    "GNU General Public License version 3.\n"
    "It is open source, free, and comes with ABSOLUTELY NO WARRANTY.\n\n\n"
    "Initialising devices...")

if args.fastboot:
    print('        Fastboot-Mode on. Devices are not calibrating.')
    print(' ')
else:
    scan_reset(1)
    scan_reset(2)
    hdmi_intro_animation()
# scan_in_thread(scan_reset, (1,))
# scan_in_thread(scan_reset, (2,))
# set_eurolite_t36(5, 0, 0, 0, 0, 0)
# set_eurolite_t36(5, 0, 0, 0, 255, 0)
set_all_pixels_color(0, 0, 0)
print("        Listening... Press Ctrl+C to stop.")
print("")
hdmi_draw_black()

# Start main loop
try:
    while True:
        # Check for user commands regarding colors, strobe and smoke effects via Redis
        redis_get_colors()
        strobe_mode = (redis_client.get('strobe_mode') or b'').decode('utf-8')
        smoke_mode = (redis_client.get('smoke_mode') or b'').decode('utf-8')
        panic_mode = (redis_client.get('panic_mode') or b'').decode('utf-8')
        play_videos = (redis_client.get('play_videos_mode') or b'').decode('utf-8')

        # Handle panic mode
        if panic_mode == 'on':
            # Turn on led strip, eurolite (floor) set on all white
            set_all_pixels_color(255, 255, 255)
            set_eurolite_t36(5, 255, 255, 255, 255, 0)
            # Blacken HDMI display
            hdmi_draw_black()
            # Turn off other lights
            scan_closed(1)
            scan_closed(2)
            while panic_mode == 'on':
                # Refresh state of panic mode
                panic_mode = (redis_client.get('panic_mode') or b'').decode('utf-8')
                send_udp_message(UDP_IP_ADDRESS, UDP_PORT, "led_45_255_255_255_255_255_255_255")
                time.sleep(0.105)
            # Restore default display after panicking
            set_all_pixels_color(0, 0, 0)
            set_eurolite_t36(5, 0, 0, 0, 255, 0)
            send_udp_message(UDP_IP_ADDRESS, UDP_PORT, "led_0_0_0_0_0_0_0_0")
            hdmi_intro_animation()
            scan_opened(1)
            scan_opened(2)

        # Handle smoke mode
        if smoke_mode == 'on':
            set_eurolite_t36(5, st_r, st_g, st_b, 255, 0)
            smoke_on()
            # set_eurolite_t36(5, 0, 0, 0, 255, 0)
        else:
            smoke_off()

        # Handle strobe mode when explicitly set to "on"
        if strobe_mode == 'on':
            # Stop any ongoing HDMI display
            kill_current_hdmi()
            # Prepare for strobing by turning off other lights
            scan_closed(1)
            scan_closed(2)
            laser_off()
            if smoke_mode != 'on':
                set_eurolite_t36(5, 0, 0, 0, 255, 0)
            # Unnecessary line as turning off lights is handled above, consider removal
            hdmi_draw_black()  # Consider removal
            while strobe_mode == 'on':
                led_strobe_effect(1, 75)
                strobe_mode = (redis_client.get('strobe_mode') or b'').decode('utf-8')
            # Restore default display after strobing
            hdmi_intro_animation()
            scan_opened(1)
            scan_opened(2)

        # Increment and manage runtime counters (consider refactoring if not needed)
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
        # Uncomment below for debugging runtime counters
        # print(runtime_mb, runtime_kb, runtime_byte, runtime_bit)

        # Read audio buffer
        audiobuffer = line_in.read(int(buffer_size / 2), exception_on_overflow=False)
        signal_input = np.frombuffer(audiobuffer, dtype=np.float32)
        signal_max = np.max(signal_input)
        # signal_mean = safe_mean(signal_input)

        # Adjust signal gain if necessary (comment suggests it's not working properly)
        signal, gain_factor = adjust_gain(volumes, signal_input)

        # Compute and store current volume
        current_volume = np.sqrt(safe_mean(signal ** 2))
        volumes.append(current_volume)

        # Compute the mean volume
        mean_volume = safe_mean(np.array(volumes))

        # Calculate the relative volume
        relative_volume = 0 if mean_volume == 0 else current_volume / mean_volume

        # Clip the value between 0 and 1
        relative_volume = min(1, max(0, relative_volume))

        # Filter and store values for low, mid, and high frequency signals
        low_signal, low_zi = sosfilt(low_sos, signal, zi=low_zi)
        low_volumes.append(np.sqrt(safe_mean(low_signal ** 2)))

        mid_signal, mid_zi = sosfilt(mid_sos, signal, zi=mid_zi)
        mid_volumes.append(np.sqrt(safe_mean(mid_signal ** 2)))

        high_signal, high_zi = sosfilt(high_sos, signal, zi=high_zi)
        high_volumes.append(np.sqrt(safe_mean(high_signal ** 2)))

        # Compute average volumes for frequency bands
        low_mean = compute_mean_volume(low_volumes)
        mid_mean = compute_mean_volume(mid_volumes)
        high_mean = compute_mean_volume(high_volumes)

        if low_mean != 0:
            low_relative = safe_mean(low_signal) / low_mean
            low_relative = min(1, max(0, low_relative))
        else:
            low_relative = 0

        # Energy calculations
        energy = np.sum(signal ** 2)
        relative_energy = energy / len(signal)

        # THX signal filtering
        thx_signal, zi = sosfilt(thx_sos, signal, zi=thx_zi)
        thx_signal = thx_signal.astype(np.float32)

        # Metrics related to "heaviness" of signal
        heavyvols.append(np.max(thx_signal))
        delta_value = np.max(thx_signal) - np.min(thx_signal)
        max_values.append(np.max(thx_signal))
        delta_values.append(delta_value)
        count_over = sum(1 for value in max_values if value > 0.08)
        last_counts.append(count_over)
        heavy, heavy_counter = is_heavy(signal, delta_values, count_over, max_values, last_counts, heavy_counter)
        heaviness = calculate_heaviness(delta_value, count_over, gain_factor, heavy_counter)
        heaviness_values.append(heaviness)

        """ New Psyco-Acoustic-Analysis """
        # LUFS-loudness-analysis
        # loudness = meter.integrated_loudness(signal)

        # Fast Fourier Transformation
        # fft_magnitude, fft_frequencies = perform_fft(signal, sample_rate)

        # Psycho acoustic weighting based on Fletcher-Munson
        # weighted_fft_magnitude = apply_fletcher_munson_curve(fft_magnitude, fft_frequencies)

        # Find dominant harmony by Fast Fourier Transformation and Psycho acoustic weighting based on ISO 226:2003
        """ shit happens here """
        # dominant_harmony = find_dominant_harmony_in_timeframe(signal, sample_rate)

        # debug
        # print(dominant_harmony)

        """ Old Acoustic Calculations """
        # Dominant frequency analysis
        dominant_freq = dominant_frequency(signal, sample_rate)
        dominant_frequencies.append(dominant_freq)
        heaviness_history.append(heavy)
        drop = detect_drop(safe_mean(volumes), heavy, dominant_frequencies, heaviness_history, drop_history)
        drop_history.append(drop)

        # Beat detection
        is_beat = aubio_onset(thx_signal)
        pitch = get_pitch(audiobuffer, pDetection)
        pitches.append(pitch)

        # Check for auto-strobe conditions and execute strobe if criteria met
        if strobe_mode == 'auto' and heavy and 1 in list(done_chase)[-10:]:
            if smoke_mode != 'on':
                set_eurolite_t36(5, 0, 0, 0, 255, 0)
            laser_off()
            kill_current_hdmi()
            scan_closed(1)
            scan_closed(2)
            hdmi_draw_black()
            led_strobe_effect(10, 75)
            hdmi_intro_animation()
            done_chase.clear()

        # Color transformations based on signal energy
        red = min(int(energy * 10), 255)
        y = max(min(((int(energy * 10) - 60) * 1.75), 255), 0)
        x = int(exponential_decrease(red))

        # DMX and LED operations
        done_chase.append(0)
        scan_gobo(1, 7, 150)  # TODO - bug: go does nothing!
        scan_gobo(2, 7, 150)  # same as above
        scan_in_thread(scan_color, (1, interpret_color(st_prim_color)))
        scan_in_thread(scan_color, (2, interpret_color(secondary_color)))

        if smoke_mode != 'on':
            set_eurolite_t36(5, x * nd_r / 255, x * nd_g / 255, x * nd_b / 255, 255, 0)  # TODO color calculation

        # send to Arduino
        udp_led = int(y / 8.5)  # for 30 LEDs
        # num_led, brightness, startR, startG, startB, endR, endG, endB
        udp_message = f"led_{udp_led}_255_{st_r}_{st_g}_{st_b}_{nd_r}_{nd_g}_{nd_b}"
        send_udp_message(UDP_IP_ADDRESS, UDP_PORT, udp_message)
        # Handle actions for heavy signal

        if heavy:
            laser_fast_dance(x, y, nd_color_name)
            hdmi_video_stop()
            no_drop_count = 0
            # led_music_visualizer_old(signal_max, st_color_name, nd_color_name)
            led_music_visualizer(low_relative, st_color_name, nd_color_name)
            scan_opened(1)
            scan_opened(2)
            scan_in_thread(scan_axis, (1, y, x))  # Front scanner
            scan_in_thread(scan_axis, (2, x, y))  # Rear scanner
            drop = False

            # Check if a beat is detected
            if is_beat:
                drop_history.clear()

            # Decrement heavy counter if it's greater than 0
            if heavy_counter > 0:
                heavy_counter -= 1

            # Generate visualization matrix based on signal
            hdmi_matrix = generate_matrix(low_signal, mid_signal, high_signal, low_mean, mid_mean, high_mean)
            transposed_hdmi_matrix = list(map(list, zip(*hdmi_matrix)))

            # Update HDMI display with computed matrix
            hdmi_draw_matrix(transposed_hdmi_matrix, st_prim_color, nd_prim_color, secondary_color)

        else:
            scan_go_home(1)
            scan_go_home(2)
            laser_slow_dance()
            if play_videos == "auto":
                video_path = "/musictolight/vids/"
                # hdmi_video_start()
                hdmi_play_video(video_path)
                if not is_video_playing():
                    # Generate visualization matrix based on signal
                    hdmi_matrix = generate_matrix(low_signal, mid_signal, high_signal, low_mean, mid_mean, high_mean)
                    transposed_hdmi_matrix = list(map(list, zip(*hdmi_matrix)))

                    # Update HDMI display with computed matrix
                    hdmi_draw_matrix(transposed_hdmi_matrix, st_prim_color, nd_prim_color, secondary_color)

            else:
                hdmi_video_stop()
                # Generate visualization matrix based on signal
                hdmi_matrix = generate_matrix(low_signal, mid_signal, high_signal, low_mean, mid_mean, high_mean)
                transposed_hdmi_matrix = list(map(list, zip(*hdmi_matrix)))

                # Update HDMI display with computed matrix
                hdmi_draw_matrix(transposed_hdmi_matrix, st_prim_color, nd_prim_color, secondary_color)

            # Handle light actions based on signal strength and history
            if signal_max > 0.007:
                input_history.append(1.0)

                if not heavy and (0 < sum(drop_history) < 32 and drop):
                    # () or (0 < sum(drop_history) < 32 and drop)
                    color_flow(runtime_bit, signal_max, 2, st_color_name, nd_color_name)
                elif not heavy:
                    color_flow(runtime_bit, signal_max, 20, st_color_name, nd_color_name)
                    no_drop_count += 1
                    if no_drop_count < 500:
                        scan_closed(1)
                        scan_closed(2)
                    else:
                        color_flow(runtime_bit, signal_max, 2, st_color_name, nd_color_name)

                # Manage animations and lights for persistent drops
                if sum(drop_history) >= 32 and drop:
                    heaviness_history.clear()
                    if 1 not in done_chase:
                        set_all_pixels_color(0, 0, 0)
                        laser_off()
                        scan_closed(1)
                        scan_closed(2)
                        hdmi_draw_black()
                        if smoke_mode != 'on':
                            set_eurolite_t36(5, st_r, st_g, st_b, 255, 0)
                        if smoke_mode == 'auto':
                            smoke_on()
                        # hdmi_outro_animation()
                        laser_star_chase()
                        star_chase(Color(127, 127, 127), 52)

                        if smoke_mode != 'on':
                            set_eurolite_t36(5, 0, 0, 0, 255, 0)
                        # hdmi_intro_animation()
                        scan_opened(1)
                        scan_opened(2)
                    # else:
                    # color_flow(runtime_bit, signal_max, 1, st_color_name, nd_color_name)  # Adjust brightness
                    done_chase.append(1)
                    smoke_off()
            else:
                input_history.append(0.0)

                # Reset state when average input is low
                if safe_mean(input_history) < 0.5:
                    drop = False
                    heavy = False
                    low_volumes.clear()
                    mid_volumes.clear()
                    high_volumes.clear()
                    pitches.clear()
                    drop_history.clear()
                    heaviness_history.clear()
                    color_flow(runtime_bit, signal_max, 10, st_color_name, nd_color_name)  # Adjust brightness
        # print(is_video_playing())


# Catch a keyboard interrupt to ensure graceful exit and cleanup
except KeyboardInterrupt:
    laser_off()
    set_all_pixels_color(0, 0, 0)  # Clear any existing colors
    hdmi_video_stop()
    # Cleanup functions to ensure a safe shutdown
    cleanup_smoke()
    hdmi_outro_animation()
    print("\nEnding program...")
    set_all_pixels_color(0, 0, 0)  # Clear any existing colors
    scan_closed(1)
    scan_closed(2)
    set_eurolite_t36(5, 0, 0, 0, 0, 0)  # Reset the eurolite device
    line_in.close()
    p.terminate()
    hdmi_video_stop()

    # Ensure any HDMI thread finishes before program exit
    if current_hdmi_thread and current_hdmi_thread.is_alive():
        current_hdmi_thread.join()

    time.sleep(2)  # Pause for 2 seconds

    # Display the license and copyright information on HDMI
    hdmi_draw_centered_text(
        "MusicToLight3  Copyright (C) 2023  Felix Rau\n\n\n"
        "This program is licensed under the terms of the \n"
        "GNU General Public License version 3.\n"
        "It is open source, free, and comes with ABSOLUTELY NO WARRANTY.\n"
        "\n\nProgram ended gracefully.")

    if not args.fastboot:
        time.sleep(5)  # Pause for 5 seconds

    # Print the license and copyright information to the console
    print("\nProgram ended gracefully.\n")
    print("MusicToLight3  Copyright (C) 2023  Felix Rau")
    print("This program is licensed under the terms of the ")
    print("GNU General Public License version 3.")
    print("It comes with ABSOLUTELY NO WARRANTY; for details see README.md.")
    print("This is free software, and you are welcome to redistribute it")
    print("under certain conditions; see LICENSE.md.\n")
