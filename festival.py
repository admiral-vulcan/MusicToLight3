# MusicToLight FESTIVAL EDITION  Copyright (C) 2025  Felix Rau.
# Licensed under the GNU GPL v3. See LICENSE for details.
#
# This program analyzes live music input, processes frequency bands,
# and visualizes them in real time for light installations or projections.
# Designed for multi-display setups with a left/right stereo concept and
# customizable opacity maps for each projector area.
#
# Perfect for festivals, art installations, and synchronized audio-visual shows!


import pygame
import sounddevice as sd
import numpy as np
import threading
import time
import os
import sys
import json
from PIL import Image
from collections import deque


# If DISPLAY not set (headless), default to :0
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

# Tile grid configuration
ROWS = 9   # number of frequency bands (vertical tiles)
COLS = 16  # number of time steps visualized horizontally (scrolling history)

# Audio settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 2
flip_channels = True
BAND_HISTORY_LEN = 60  # normalization over about 2 seconds at 30 FPS
OFFSET_CORRECTION = 0.015

# Channel selection: set in __main__
SELECTED_CHANNEL = None

# File paths
OFFSET_FILE = "audio_offset.json"
OPACITY_MAP_LEFT = "left_op.png"
OPACITY_MAP_RIGHT = "right_op.png"

# Buffers and gains
audio_levels = np.zeros(2)
audio_buffer = np.zeros(BUFFER_SIZE * 4)
#band_gains = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 3.1, 5.2, 7.6, 30])
band_gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
boost = 40

# Colors for visualization
COLOR_TREBLE = (0, 160, 255)
COLOR_MID = (255, 255, 40)
COLOR_BASS = (255, 0, 32)

# Audio threading
audio_running = threading.Event()


def print_intro():
    print(r"""
╔════════════════════════════════════════════════════════════════╗
║    MusicToLight FESTIVAL EDITION                               ║
║    Copyright (C) 2025 Felix Rau - Licensed under GNU GPL v3    ║
╚════════════════════════════════════════════════════════════════╝

This program analyzes live music input, processes frequency bands,
and visualizes them in real time for light installations or projections.
Designed for multi-display setups with a left/right stereo concept and
customizable opacity maps for each projector area.

Perfect for festivals, art installations, and synchronized audio-visual shows!

""")


def print_usage():
    print("""
Usage: python3 festival.py [left|right] [mapping] [fullgrid]
       python3 festival.py calibrate

Arguments:
  left        Visualize the left audio channel (for the left display).
  right       Visualize the right audio channel (for the right display).
  mapping     (Optional) Show the opacity map for projector alignment and mask verification.
  fullgrid    (Optional) Display the complete grid of tiles, ignoring any opacity map.
  calibrate   Start the audio offset and noise-floor calibration. Run this in silence.

Notes:
- Either 'left' or 'right' must be specified.
- 'mapping' and/or 'fullgrid' can be added in any order, together with 'left' or 'right'.
- Example: python3 festival.py left mapping
           python3 festival.py mapping right
           python3 festival.py left fullgrid

If no valid mode is specified, this help is shown.
""")


def set_display_position(selected_channel):
    if selected_channel == 'left':
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    elif selected_channel == 'right':
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1280,0"


def lerp(a, b, t):
    """Linear interpolation between colors a and b, t in [0, 1]."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def get_band_color(row, bands_n):
    """Returns a color for a frequency band (row), blending from treble to bass."""
    if bands_n < 3:
        t = row / (bands_n - 1)
        return lerp(COLOR_TREBLE, COLOR_BASS, t)
    mid_row = (bands_n - 1) / 2
    if row <= mid_row:
        t = row / mid_row
        return lerp(COLOR_TREBLE, COLOR_MID, t)
    else:
        t = (row - mid_row) / (bands_n - 1 - mid_row)
        return lerp(COLOR_MID, COLOR_BASS, t)


def audio_callback(indata, frames, time_info, status):
    """
    Audio callback function for sounddevice.InputStream.
    Processes either one selected channel ('left' or 'right') or both (in 'both' mode for calibration).
    Updates global audio_levels and audio_buffer accordingly.
    """
    global audio_levels, audio_buffer, SELECTED_CHANNEL

    if SELECTED_CHANNEL == 'both':
        # Calibration mode: process both left and right channels
        if flip_channels:
            left = indata[:, 1]
            right = indata[:, 0]
        else:
            left = indata[:, 0]
            right = indata[:, 1]

        # Calculate RMS levels for both channels
        level_left = np.sqrt(np.mean(left ** 2))
        level_right = np.sqrt(np.mean(right ** 2))
        audio_levels = [level_left, level_right]

        # Create mono mix for spectral analysis
        mono = (left + right) / 2
        audio_buffer[:-frames] = audio_buffer[frames:]
        audio_buffer[-frames:] = mono

    else:
        # Single channel mode: process only selected channel
        if SELECTED_CHANNEL == 'left':
            idx = 1 if flip_channels else 0
        else:  # 'right'
            idx = 0 if flip_channels else 1

        channel_data = indata[:, idx]

        # Calculate RMS level for the selected channel
        level = np.sqrt(np.mean(channel_data ** 2))
        audio_levels = [level, 0.0]  # Keep format consistent for downstream code

        # Update buffer for spectral analysis
        audio_buffer[:-frames] = audio_buffer[frames:]
        audio_buffer[-frames:] = channel_data


def audio_thread():
    """Starts the audio input stream in a separate thread."""
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=BUFFER_SIZE,
        callback=audio_callback
    ):
        while audio_running.is_set():
            time.sleep(0.01)


def calibrate(seconds=5):
    """Runs calibration for offset and band noise floor. Must be silent."""
    if SELECTED_CHANNEL != 'both':
        print(f"Calibration can only be run in stereo mode.")
        sys.exit(0)
    print(f"Starting calibration ({seconds} seconds silence, do not input any sound)...")
    levels_left = []
    levels_right = []
    band_history = []

    # Start audio thread
    audio_running.set()
    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()

    pygame.init()
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    win = pygame.display.set_mode((1280, 720), pygame.NOFRAME)
    pygame.display.set_caption("Calibration: Silence Please!")
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("monospace", 120)
    clock = pygame.time.Clock()
    for sec in reversed(range(1, seconds + 1)):
        for _ in range(30):
            win.fill((0, 0, 0))
            msg = font.render(f"Calibrating: {sec}", True, (255, 255, 0))
            win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
            pygame.display.flip()
            clock.tick(30)
            bands = compute_bands(audio_buffer, SAMPLE_RATE, bands_n=ROWS, band_noise_floor=None, normalize=False)
            band_history.append(bands)
        levels_left.append(audio_levels[0])
        levels_right.append(audio_levels[1])

    win.fill((0, 0, 0))
    msg = font.render("Done!", True, (0, 255, 0))
    win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
    pygame.display.flip()
    time.sleep(1)
    audio_running.clear()
    t_audio.join()
    pygame.quit()

    offset_left = float(np.median(levels_left))
    offset_right = float(np.median(levels_right))
    offset_data = {"left": offset_left, "right": offset_right}
    with open(OFFSET_FILE, "w") as f:
        json.dump(offset_data, f, indent=2)
    print(f"Calibration complete! Values saved to {OFFSET_FILE}: {offset_data}")

    band_history = np.array(band_history)
    band_noise = band_history.mean(axis=0)
    band_noise_file = "band_noise_floor.json"
    with open(band_noise_file, "w") as f:
        json.dump(band_noise.tolist(), f, indent=2)
    print(f"Band noise floor saved to {band_noise_file}: {band_noise}")


def load_offset():
    """Loads the offset from file, returns [left, right] or None."""
    try:
        with open(OFFSET_FILE, "r") as f:
            d = json.load(f)
        common_offset = max(float(d["left"]), float(d["right"]))
        return [common_offset, common_offset]
    except Exception:
        return None


def load_band_noise_floor(bands_n=ROWS):
    """Loads the noise floor for each frequency band."""
    try:
        with open("band_noise_floor.json", "r") as f:
            arr = np.array(json.load(f))
        if arr.shape[0] != bands_n:
            print("Warning: Noise floor does not match band count!")
        return arr
    except Exception:
        print("No band noise floor found, using zeros.")
        return np.zeros(bands_n)


def load_mask_surface(mask_path, resolution):
    """Loads a grayscale PNG as alpha mask, resizes to screen resolution."""
    img = Image.open(mask_path).convert("L").resize(resolution)
    alpha_array = np.array(img)
    print("Mask shape:", alpha_array.shape, "min/max:", alpha_array.min(), alpha_array.max())
    mask_surface = pygame.Surface(resolution, pygame.SRCALPHA)
    mask_pixels = pygame.surfarray.pixels_alpha(mask_surface)
    mask_pixels[:, :] = alpha_array.T
    del mask_pixels
    rgb = pygame.surfarray.pixels3d(mask_surface)
    rgb[:, :, :] = 255
    del rgb
    return mask_surface


def build_tilemap(mask_path, size, rows=ROWS, cols=COLS, threshold=128):
    """Creates a tile map based on the opacity map and desired grid size."""
    opacity_map = np.array(Image.open(mask_path).convert("L").resize(size))
    cell_h = size[1] // rows
    cell_w = size[0] // cols
    tiles = []
    for row in range(rows):
        for col in range(cols):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            if np.max(opacity_map[y1:y2, x1:x2]) > threshold:
                tiles.append({
                    "x": x1, "y": y1, "w": cell_w, "h": cell_h, "row": row, "col": col
                })
    return tiles


def compute_bands(audio, sample_rate, bands_n=ROWS, band_noise_floor=None, normalize=True):
    """
    FFT-based frequency band extraction and dynamic normalization.
    Uses a rolling max of each band over the last N frames for adaptive normalization.
    """
    fft = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    band_edges = np.logspace(np.log10(40), np.log10(sample_rate / 2), bands_n + 1)
    band_values = []
    for i in range(bands_n):
        idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i + 1]))[0]
        if len(idx) > 0:
            band_values.append(np.sqrt(np.mean(fft[idx] ** 2)))
        else:
            band_values.append(0.0)
    band_values = np.array(band_values)

    if band_noise_floor is not None:
        band_values = np.maximum(band_values - band_noise_floor, 0)

    if normalize:
        # -------- Dynamic normalization using a rolling maximum over N frames --------
        # Threshold small values for stability (using calibrated offset)
        band_values[band_values < OFFSET_CORRECTION] = 0

        # Initialize per-band rolling max buffers on first run
        if not hasattr(compute_bands, "band_max_buffers") or len(compute_bands.band_max_buffers) != bands_n:
            from collections import deque
            # Create one deque per band for the last N values
            compute_bands.band_max_buffers = [deque([1e-6] * BAND_HISTORY_LEN, maxlen=BAND_HISTORY_LEN) for _ in
                                              range(bands_n)]
        band_max_buffers = compute_bands.band_max_buffers

        dynamic_max = np.zeros(bands_n)
        for i in range(bands_n):
            # Only update the rolling history if the value is > 0 (otherwise keep last real value)
            if band_values[i] > 0:
                band_max_buffers[i].append(band_values[i])
            # Use max of last N frames for normalization
            dynamic_max[i] = max(band_max_buffers[i])
            # Prevent division by zero (if the buffer is all zeros)
            if dynamic_max[i] < 1e-8:
                dynamic_max[i] = 1.0

        # Normalize bands individually to their recent maximum
        band_values = band_values / dynamic_max


    # Apply gain correction per band
    band_values = band_values * band_gains[:bands_n]
    # Clamp to [0, 1]
    band_values = np.clip(band_values, 0, 1)

    return band_values


def show_mapping_preview(alpha_mask, size):
    """Displays the mapping overlay fullscreen with label."""
    pygame.init()
    win = pygame.display.set_mode(size, pygame.FULLSCREEN)
    pygame.display.set_caption("Mapping Preview")
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("monospace", 60)
    clock = pygame.time.Clock()
    test_surf = pygame.Surface(size, pygame.SRCALPHA)
    test_surf.fill((255, 255, 255, 255))
    test_surf.blit(alpha_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    info_text = f"Mapping: {SELECTED_CHANNEL.upper()}"
    text_color = (255, 220, 60)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
        win.fill((0, 0, 0))
        win.blit(test_surf, (0, 0))
        msg = font.render(info_text, True, text_color)
        win.blit(msg, (size[0] // 2 - msg.get_width() // 2, 20))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def main(mapping=False):
    """
    Main loop for MusicToLight visualization.
    If mapping is True, only show the opacity mask.
    If SELECTED_CHANNEL is 'left' or 'right', visualize only the respective audio channel.
    """
    global offset

    # Load calibrated audio offset from file (used to remove input noise floor)
    offset = load_offset()
    # Load per-band noise floor values from file (for improved frequency normalization)
    band_noise_floor = load_band_noise_floor(bands_n=ROWS)
    # Buffer to store previous row values (used for jitter detection and effect)
    prev_row_values = np.zeros((ROWS, COLS), dtype=float)

    if offset is None:
        # Show error and exit if no calibration data is available
        print(f"Offset file '{OFFSET_FILE}' is missing! Please run calibration first:\n   python3 {sys.argv[0]} calibrate")
        pygame.init()
        info = pygame.display.Info()
        size = (info.current_w, info.current_h)
        win = pygame.display.set_mode(size, pygame.FULLSCREEN)
        pygame.display.set_caption("MusicToLight Level Display")
        pygame.mouse.set_visible(False)
        font = pygame.font.SysFont("monospace", 80)
        win.fill((0, 0, 0))
        msg = font.render("Offset missing!", True, (255, 0, 0))
        win.blit(msg, (size[0]//2 - msg.get_width()//2, size[1]//2 - msg.get_height()//2))
        pygame.display.flip()
        time.sleep(4)
        pygame.quit()
        return

    # Start audio thread
    audio_running.set()
    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()

    # Initialize Pygame display
    pygame.init()
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    win = pygame.display.set_mode(size, pygame.FULLSCREEN)
    pygame.display.set_caption("MusicToLight Level Display")
    pygame.mouse.set_visible(False)

    # Choose mask based on SELECTED_CHANNEL
    if SELECTED_CHANNEL == 'right':
        mask_path = os.path.join(os.path.dirname(__file__), OPACITY_MAP_RIGHT)
    else:
        mask_path = os.path.join(os.path.dirname(__file__), OPACITY_MAP_LEFT)

    if fullgrid_mode:
        # Grid without opacity map
        print("Full grid mode: Showing all tiles!")
        tiles = []
        cell_h = size[1] // ROWS
        cell_w = size[0] // COLS
        for row in range(ROWS):
            for col in range(COLS):
                x = col * cell_w
                y = row * cell_h
                tiles.append({
                    "x": x, "y": y, "w": cell_w, "h": cell_h, "row": row, "col": col
                })
        scroll_values = np.zeros((ROWS, COLS), dtype=float)
        alpha_mask = None
    else:
        # Grid with opacity map
        if not os.path.isfile(mask_path):
            alpha_mask = None
            tiles = []
            print("Opacity map not found:", mask_path)
        else:
            alpha_mask = load_mask_surface(mask_path, size)
            print("Opacity map loaded:", mask_path)
            tiles = build_tilemap(mask_path, size)
            scroll_values = np.zeros((ROWS, COLS), dtype=float)
            print(f"{len(tiles)} of {ROWS * COLS} tiles visible in this map.")
            if mapping and alpha_mask:
                show_mapping_preview(alpha_mask, size)
                audio_running.clear()
                t_audio.join()
                return

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Compute loudness and visualization parameters
            loud = max(audio_levels[0] - offset[0], 0.0)
            loud_vis = min(loud * boost, 1.0)
            color = (int(loud_vis * 255), 0, int((1 - loud_vis) * 128) + 64)

            # FFT bands
            current_audio = audio_buffer
            band_values = compute_bands(current_audio, SAMPLE_RATE, bands_n=ROWS, band_noise_floor=band_noise_floor)

            # Scroll effect: leftmost (or rightmost) tile is current value, values move outward
            for row in range(ROWS):
                if SELECTED_CHANNEL == "left":
                    # left channel: new value leftmost, shift right
                    for col in reversed(range(1, COLS)):
                        scroll_values[row, col] = scroll_values[row, col - 1]
                    scroll_values[row, 0] = band_values[len(band_values) - 1 - row]
                elif SELECTED_CHANNEL == "right":
                    # right channel: new value rightmost, shift left
                    for col in range(COLS - 1):
                        scroll_values[row, col] = scroll_values[row, col + 1]
                    scroll_values[row, COLS - 1] = band_values[len(band_values) - 1 - row]
                else:
                    # fallback: left to right
                    for col in reversed(range(1, COLS)):
                        scroll_values[row, col] = scroll_values[row, col - 1]
                    scroll_values[row, 0] = band_values[len(band_values) - 1 - row]

            # Row-wise jitter if a row remains static, to keep visuals lively
            for row in range(ROWS):
                # If the entire row has not changed since the last frame, apply jitter
                if np.allclose(scroll_values[row, :], prev_row_values[row, :], atol=1e-6):
                    # Add a small random noise to each tile in the row to create a subtle movement effect
                    scroll_values[row, :] += np.random.uniform(-0.03, 0.03, size=COLS)
                    # Clamp values to valid range [0, 1] after jitter
                    scroll_values[row, :] = np.clip(scroll_values[row, :], 0, 1)
            # Store the current row values for comparison in the next frame
            prev_row_values[:, :] = scroll_values[:, :]

            # Draw visualization
            vis_surface = pygame.Surface(size, pygame.SRCALPHA)
            for k in tiles:
                row = k["row"]
                col = k["col"]
                val = scroll_values[row, col]
                scale = 0.1 + val * 0.85
                w = int(k["w"] * scale)
                h = int(k["h"] * scale)
                x = k["x"] + (k["w"] - w) // 2
                y = k["y"] + (k["h"] - h) // 2
                color = get_band_color(row, ROWS)
                pygame.draw.rect(vis_surface, color, (x, y, w, h))

            win.fill((0, 0, 0))
            win.blit(vis_surface, (0, 0))
            pygame.display.flip()

            clock.tick(30)
    finally:
        audio_running.clear()
        t_audio.join()
        pygame.quit()


if __name__ == '__main__':
    print_intro()
    args = set(arg.lower() for arg in sys.argv[1:])

    # Set SELECTED_CHANNEL and start main
    if 'right' in args:
        SELECTED_CHANNEL = 'right'
    else:
        SELECTED_CHANNEL = 'left'
    set_display_position(SELECTED_CHANNEL)

    # Validate args
    if 'calibrate' in args:
        SELECTED_CHANNEL = 'both'
        calibrate()
        sys.exit(0)
    if not (('left' in args) ^ ('right' in args)):
        print_usage()
        print("\nYou must specify either 'left' or 'right' as the display side. Optionally add 'mapping' for mask preview.")
        sys.exit(0)

    mapping = 'mapping' in args
    fullgrid_mode = 'fullgrid' in args

    try:
        main(mapping=mapping)
    except KeyboardInterrupt:
        print("\n\nProgram ended.")
