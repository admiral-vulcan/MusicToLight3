# MusicToLight FESTIVAL EDITION Dual Monitor Update (2025)
# Copyright (C) 2025  Felix Rau
# Licensed under the GNU GPL v3. See LICENSE for details.
#
# This program analyzes live music input, processes frequency bands,
# and visualizes them in real time for light installations or projections.
# Designed for multi-display setups with a left/right stereo concept and
# customizable opacity maps for each projector area.
#
# Perfect for festivals, art installations, and synchronized audio-visual shows!
#
# For problems running on two monitors see DUALMONITOR.md and run set_hdmi_resolution as X11 - active user.

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

os.system('/musictolight/set_hdmi_resolutions.sh')

# Performance setting
FPS = 60  # Default is 60 fps, change if screens or Pi are performing poorly

# Dual-monitor dimensions
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WIDTH = SCREEN_WIDTH * 2       # two screens side by side
HEIGHT = SCREEN_HEIGHT - 1     # must be <> native height

# SDL/pygame hacks
os.environ['SDL_VIDEO_WINDOW_POS']       = '0,0'
os.environ['SDL_VIDEO_CENTERED']         = '0'
os.environ['SDL_VIDEO_FULLSCREEN_HEAD']  = '0'

# Tile grid configuration
ROWS = 9   # number of frequency bands (vertical tiles)
COLS = 16  # number of time steps visualized horizontally (scrolling history)

# Audio settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 2
flip_channels = True
BAND_HISTORY_LEN = 5 * FPS  # seconds times FPS
OFFSET_CORRECTION = 0.015

# Channel selection: set in __main__
SELECTED_CHANNEL = None

# File paths
OFFSET_FILE = "audio_offset.json"
OPACITY_MAP_LEFT = "left_op.png"
OPACITY_MAP_RIGHT = "right_op.png"

# Buffers and gains
audio_levels = np.zeros(2)
audio_buffer = np.zeros((BUFFER_SIZE * 4, 2))  # Stereo
#band_gains = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 3.1, 5.2, 7.6, 30])
band_gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
boost = 40

# Colors for visualization
COLOR_TREBLE = (0, 160, 255)
COLOR_MID = (255, 255, 40)
COLOR_BASS = (255, 0, 32)

# Audio threading
audio_running = threading.Event()

# Opacity map setting
ALPHA_THRESHOLD = 32 / 255  # about 12% brightness

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

    if indata.shape[1] == 2:
        l = indata[:, 0]
        r = indata[:, 1]
        audio_levels = [np.sqrt(np.mean(l ** 2)), np.sqrt(np.mean(r ** 2))]
        # Schiebe Buffer (Rolling)
        audio_buffer[:-frames, :] = audio_buffer[frames:, :]
        audio_buffer[-frames:, 0] = l
        audio_buffer[-frames:, 1] = r

    else:
        # Fallback: Mono oder nur 1 Kanal (shouldn't happen)
        audio_buffer[:-frames, 0] = audio_buffer[frames:, 0]
        audio_buffer[-frames:, 0] = indata[:, 0]
        audio_buffer[:, 1] = 0  # Setze rechten Kanal auf 0
        audio_levels = [np.sqrt(np.mean(indata[:, 0] ** 2)), 0.0]


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

    # Schwellenwert für 'zu laut' (RMS-Level, konservativ, kann angepasst werden)
    MAX_ALLOWED_RMS = 0.04

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
    aborted = False
    for sec in reversed(range(1, seconds + 1)):
        for _ in range(30):
            # --- Plausibilitätsprüfung: Ist ein Kanal zu laut?
            if max(audio_levels[0], audio_levels[1]) > MAX_ALLOWED_RMS:
                # Fenster rot färben und abbrechen
                win.fill((180, 0, 0))
                msg = font.render("CALIBRATION FAILED", True, (255, 255, 0))
                msg2 = font.render("TOO LOUD!", True, (255, 255, 0))
                win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() - 40))
                win.blit(msg2, (size[0] // 2 - msg2.get_width() // 2, size[1] // 2 + 20))
                pygame.display.flip()
                print(f"\n[ABORT] Calibration aborted: Signal too loud! (L={audio_levels[0]:.3f} / R={audio_levels[1]:.3f})")
                time.sleep(3)
                aborted = True
                break

            win.fill((0, 0, 0))
            msg = font.render(f"Calibrating: {sec}", True, (255, 255, 0))
            win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
            pygame.display.flip()
            clock.tick(FPS)

            # Band-Noise-Floor: Mittelwert aus beiden Kanälen (Kompatibilität)
            if audio_buffer.shape[1] == 2:
                avg = np.mean(audio_buffer, axis=1)
            else:
                avg = audio_buffer[:, 0]
            bands = compute_bands(avg, SAMPLE_RATE, bands_n=ROWS, band_noise_floor=None, normalize=False)
            band_history.append(bands)
        if aborted:
            break
        levels_left.append(audio_levels[0])
        levels_right.append(audio_levels[1])

    audio_running.clear()
    t_audio.join()

    if aborted:
        pygame.quit()
        print("Calibration failed: Please ensure no audio is playing and try again.")
        return

    win.fill((0, 0, 0))
    msg = font.render("Done!", True, (0, 255, 0))
    win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
    pygame.display.flip()
    time.sleep(1)
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
        clock.tick(FPS)
    pygame.quit()


def main_dual(fullgrid_mode=False, mapping=False):
    """
    Dual-Monitor mode:
    - No args: dual output (left visuals on left, right visuals on right)
    - fullgrid_mode: ignore opacity maps, all tiles shown
    - mapping: preview both opacity maps side by side, then exit
    """
    # --- Load calibration and noise floor ---
    offset = load_offset()
    band_noise_floor = load_band_noise_floor(bands_n=ROWS)

    # --- Start audio thread ---
    audio_running.set()
    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()

    # --- Pygame window setup (2560×719 for dual monitor hack) ---
    pygame.init()
    size = (WIDTH, HEIGHT)
    win = pygame.display.set_mode(size, pygame.NOFRAME)
    pygame.display.set_caption("MusicToLight Dual-Monitor")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # --- Prepare tile maps and mask alphas for each side ---
    left_tiles, right_tiles = [], []

    if fullgrid_mode:
        # All tiles shown, alpha always 1.0
        cw, ch = SCREEN_WIDTH // COLS, HEIGHT // ROWS
        for r in range(ROWS):
            for c in range(COLS):
                left_tiles.append({"row": r, "col": c,
                                   "x": c * cw,
                                   "y": r * ch,
                                   "w": cw, "h": ch,
                                   "alpha": 1.0})
                right_tiles.append({"row": r, "col": c,
                                    "x": SCREEN_WIDTH + c * cw,
                                    "y": r * ch,
                                    "w": cw, "h": ch,
                                    "alpha": 1.0})
    else:
        # Opacity-maps: load as numpy arrays
        mask_left_path = os.path.join(os.path.dirname(__file__), OPACITY_MAP_LEFT)
        mask_right_path = os.path.join(os.path.dirname(__file__), OPACITY_MAP_RIGHT)

        mask_left_img = Image.open(mask_left_path).convert("L").resize((SCREEN_WIDTH, HEIGHT))
        mask_right_img = Image.open(mask_right_path).convert("L").resize((SCREEN_WIDTH, HEIGHT))
        mask_left_np = np.array(mask_left_img)
        mask_right_np = np.array(mask_right_img)

        # --- Print mask info (Debug/Status) ---
        left_min, left_max = mask_left_np.min(), mask_left_np.max()
        right_min, right_max = mask_right_np.min(), mask_right_np.max()
        print(f"[HDMI-1] Mask shape: {mask_left_np.shape} min/max: {left_min} {left_max}")
        print(f"[HDMI-1] Opacity map loaded: {mask_left_path}")

        print(f"[HDMI-2] Mask shape: {mask_right_np.shape} min/max: {right_min} {right_max}")
        print(f"[HDMI-2] Opacity map loaded: {mask_right_path}")

        # Build tiles with precomputed per-tile alpha
        cw, ch = SCREEN_WIDTH // COLS, HEIGHT // ROWS
        for r in range(ROWS):
            for c in range(COLS):
                # Left screen
                x = c * cw
                y = r * ch
                # Take alpha as mean over tile area (or just center pixel, up to you!)
                a_tile = np.mean(mask_left_np[y:y+ch, x:x+cw]) / 255.0
                left_tiles.append({"row": r, "col": c, "x": x, "y": y, "w": cw, "h": ch, "alpha": a_tile})
                # Right screen
                x2 = c * cw
                y2 = r * ch
                a2_tile = np.mean(mask_right_np[y2:y2+ch, x2:x2+cw]) / 255.0
                right_tiles.append({"row": r, "col": c,
                                    "x": SCREEN_WIDTH + x2, "y": y2, "w": cw, "h": ch, "alpha": a2_tile})

        print(f"[HDMI-1] {sum(1 for t in left_tiles if t['alpha'] >= ALPHA_THRESHOLD)} of {ROWS * COLS} tiles visible in this map.")
        print(f"[HDMI-2] {sum(1 for t in right_tiles if t['alpha'] >= ALPHA_THRESHOLD)} of {ROWS * COLS} tiles visible in this map.")

    # --- Mapping preview: show both maps side by side and exit ---
    if mapping and not fullgrid_mode:
        preview = pygame.Surface(size, pygame.SRCALPHA)
        left_img = pygame.image.fromstring(mask_left_img.tobytes(), mask_left_img.size, "L").convert()
        right_img = pygame.image.fromstring(mask_right_img.tobytes(), mask_right_img.size, "L").convert()
        preview.blit(left_img, (0, 0))
        preview.blit(right_img, (SCREEN_WIDTH, 0))
        running = True
        while running:
            for e in pygame.event.get():
                if e.type in (pygame.QUIT, pygame.KEYDOWN):
                    running = False
            win.fill((0, 0, 0))
            win.blit(preview, (0, 0))
            pygame.display.flip()
            clock.tick(FPS)
        audio_running.clear()
        t_audio.join()
        pygame.quit()
        return

    # --- Visualization buffers ---
    scroll_left = np.zeros((ROWS, COLS), dtype=float)
    scroll_right = np.zeros((ROWS, COLS), dtype=float)
    prev_left = np.zeros_like(scroll_left)
    prev_right = np.zeros_like(scroll_right)

    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            # Compute bands for both channels (left/right)
            # Split audio_buffer into channels for analysis
            # -> audio_callback must fill both channels!
            # For simplicity: in this version, we use only mono,
            # but you can split your buffer if you keep a stereo buffer!

            # Assume stereo is interleaved in audio_buffer
            audio_stereo = audio_buffer.reshape(-1, 2)
            if flip_channels:
                l_channel = audio_stereo[:, 1]
                r_channel = audio_stereo[:, 0]
            else:
                l_channel = audio_stereo[:, 0]
                r_channel = audio_stereo[:, 1]

            bands_left = compute_bands(l_channel, SAMPLE_RATE, bands_n=ROWS, band_noise_floor=band_noise_floor)
            bands_right = compute_bands(r_channel, SAMPLE_RATE, bands_n=ROWS, band_noise_floor=band_noise_floor)

            # Scroll effect for both sides
            for row in range(ROWS):
                # LEFT: scroll right, insert newest leftmost
                scroll_left[row, 1:] = scroll_left[row, :-1]
                scroll_left[row, 0] = bands_left[ROWS - 1 - row]
                # RIGHT: scroll left, insert newest rightmost
                scroll_right[row, :-1] = scroll_right[row, 1:]
                scroll_right[row, -1] = bands_right[ROWS - 1 - row]

            # Jitter for both (optional)
            for row in range(ROWS):
                if np.allclose(scroll_left[row], prev_left[row], atol=1e-6):
                    scroll_left[row] += np.random.uniform(-0.03, 0.03, size=COLS)
                    scroll_left[row] = np.clip(scroll_left[row], 0, 1)
                if np.allclose(scroll_right[row], prev_right[row], atol=1e-6):
                    scroll_right[row] += np.random.uniform(-0.03, 0.03, size=COLS)
                    scroll_right[row] = np.clip(scroll_right[row], 0, 1)
            prev_left[:, :] = scroll_left[:, :]
            prev_right[:, :] = scroll_right[:, :]

            # --- Draw everything ---
            win.fill((0, 0, 0))

            # Left screen tiles
            for t in left_tiles:
                if t["alpha"] < ALPHA_THRESHOLD:
                    continue  # Skip transparent tiles
                row, col = t["row"], t["col"]
                val = scroll_left[row, col]
                scale = 0.1 + val * 0.85
                w = int(t["w"] * scale)
                h = int(t["h"] * scale)
                x = t["x"] + (t["w"] - w) // 2
                y = t["y"] + (t["h"] - h) // 2
                color = get_band_color(row, ROWS)
                pygame.draw.rect(win, color, (x, y, w, h))

            # Right screen tiles
            for t in right_tiles:
                if t["alpha"] < ALPHA_THRESHOLD:
                    continue
                row, col = t["row"], t["col"]
                val = scroll_right[row, col]
                scale = 0.1 + val * 0.85
                w = int(t["w"] * scale)
                h = int(t["h"] * scale)
                x = t["x"] + (t["w"] - w) // 2
                y = t["y"] + (t["h"] - h) // 2
                color = get_band_color(row, ROWS)
                pygame.draw.rect(win, color, (x, y, w, h))

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        audio_running.clear()
        t_audio.join()
        pygame.quit()


def main_single(mapping=False, fullgrid_mode=False):
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
            channel_idx = 1 if SELECTED_CHANNEL == "left" else 0
            current_audio = audio_buffer[:, channel_idx]
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

            clock.tick(FPS)
    finally:
        audio_running.clear()
        t_audio.join()
        pygame.quit()


if __name__ == '__main__':
    print_intro()
    args = set(arg.lower() for arg in sys.argv[1:])

    # Calibration mode
    if 'calibrate' in args:
        SELECTED_CHANNEL = 'both'
        try:
            calibrate()
        except KeyboardInterrupt:
            print("\nProgram ended.")
        sys.exit(0)

    # Determine flags
    fullgrid_mode = 'fullgrid' in args
    mapping      = 'mapping' in args
    left         = 'left' in args
    right        = 'right' in args

    try:
        # Single-screen mode: exactly one of 'left' or 'right'
        if left ^ right:
            SELECTED_CHANNEL = 'left' if left else 'right'
            set_display_position(SELECTED_CHANNEL)
            main_single(mapping=mapping, fullgrid_mode=fullgrid_mode)

        # Dual-screen mode: no args OR mapping-only OR fullgrid-only OR mapping+fullgrid
        else:
            main_dual(fullgrid_mode=fullgrid_mode, mapping=mapping)

    except KeyboardInterrupt:
        print("\n\nProgram ended.")
