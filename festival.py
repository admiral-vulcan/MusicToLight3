# MusicToLight FESTIVAL EDITION  Copyright (C) 2025  Felix Rau.
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

# This is a standalone version

import pygame
import sounddevice as sd
import numpy as np
import threading
import time
import os
import sys
import json
from PIL import Image

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 2
flip_channels = True
OFFSET_FILE = "audio_offset.json"
OPACITY_MAP_FILE = "left_op.png"
audio_levels = [0.0, 0.0]
offset = [0.0, 0.0]
mapping = False
boost = 40  # 0.02 * 40 = 0.8 → bei deinem Maximalpegel ist fast volle Farbe erreicht!
audio_buffer = np.zeros(BUFFER_SIZE * 4)  # z.B. 4096 Samples
band_gains = np.array([0.5, 0.7, 0.8, 1.0, 1.2, 3.1, 5.2, 7.6, 30]) #links Tiefen, rechts Höhen

# Farben in RGB (0-255)
COLOR_TREBLE   = (0, 160, 255)   # Oben: Blau
COLOR_MID    = (255, 255, 40)  # Mitte: Gelb
COLOR_BASS = (255, 0, 32)    # Unten: Rot

def lerp(a, b, t):
    # Interpoliert zwischen zwei Farben a, b mit Faktor t (0...1)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def get_band_color(row, bands_n):
    # row: 0 = unten (Bässe), bands_n-1 = oben (Höhen)
    if bands_n < 3:
        # fallback auf Bass->Treble
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
    global audio_levels, audio_buffer
    if flip_channels:
        left = indata[:, 1]
        right = indata[:, 0]
    else:
        left = indata[:, 0]
        right = indata[:, 1]
    level_left = np.sqrt(np.mean(left**2))
    level_right = np.sqrt(np.mean(right**2))
    audio_levels = [level_left, level_right]
    # Shift: Alte Samples raus, neue rein
    mono = (left + right) / 2
    audio_buffer[:-frames] = audio_buffer[frames:]
    audio_buffer[-frames:] = mono


def audio_thread():
    with sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            callback=audio_callback):
        while True:
            time.sleep(0.01)

def calibrate(seconds=5):
    print("Starte Kalibrierung ({} Sekunden Stille, bitte nichts einspeisen)...".format(seconds))
    levels_left = []
    levels_right = []
    band_history = []

    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()
    pygame.init()
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    win = pygame.display.set_mode(size, pygame.FULLSCREEN)
    pygame.display.set_caption("Kalibrierung: Bitte Stille!")
    pygame.mouse.set_visible(False)
    font = pygame.font.SysFont("monospace", 120)
    clock = pygame.time.Clock()
    for sec in reversed(range(1, seconds + 1)):
        for _ in range(30):
            win.fill((0, 0, 0))
            msg = font.render(f"Kalibriere: {sec}", True, (255, 255, 0))
            win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
            pygame.display.flip()
            clock.tick(30)
            # Sammle Bandpegel live aus aktuellem Audio-Buffer
            bands = compute_bands(audio_buffer, SAMPLE_RATE, bands_n=9, band_noise_floor=None, normalize=False)
            band_history.append(bands)
        levels_left.append(audio_levels[0])
        levels_right.append(audio_levels[1])

    win.fill((0, 0, 0))
    msg = font.render("Fertig!", True, (0, 255, 0))
    win.blit(msg, (size[0] // 2 - msg.get_width() // 2, size[1] // 2 - msg.get_height() // 2))
    pygame.display.flip()
    time.sleep(1)
    pygame.quit()
    offset_left = float(np.median(levels_left))
    offset_right = float(np.median(levels_right))
    offset_data = {"left": offset_left, "right": offset_right}
    with open(OFFSET_FILE, "w") as f:
        json.dump(offset_data, f, indent=2)
    print("Kalibrierung erfolgreich! Werte gespeichert in {}: {}".format(OFFSET_FILE, offset_data))

    # Jetzt Noise-Floor pro Band abspeichern!
    band_history = np.array(band_history)  # (anz_samples, bands_n)
    band_noise = band_history.mean(axis=0)
    band_noise_file = "band_noise_floor.json"
    with open(band_noise_file, "w") as f:
        json.dump(band_noise.tolist(), f, indent=2)
    print("Band-Noise-Floor gespeichert in", band_noise_file, ":", band_noise)


def load_offset():
    try:
        with open(OFFSET_FILE, "r") as f:
            d = json.load(f)
        # Nimm den höheren Offset für beide Kanäle
        common_offset = max(float(d["left"]), float(d["right"]))
        return [common_offset, common_offset]
    except Exception:
        return None

def load_band_noise_floor(bands_n=9):
    try:
        with open("band_noise_floor.json", "r") as f:
            arr = np.array(json.load(f))
        if arr.shape[0] != bands_n:
            print("Warnung: Noise-Floor passt nicht zur Bandzahl!")
        return arr
    except Exception:
        print("Kein Noise-Floor pro Band gefunden, nehme 0!")
        return np.zeros(bands_n)

def load_mask_surface(mask_path, resolution):
    # Lade PNG als Graustufen, skaliere auf die Bildschirmgröße
    img = Image.open(mask_path).convert("L").resize(resolution)
    alpha_array = np.array(img)
    print("Map shape:", alpha_array.shape, "min/max:", alpha_array.min(), alpha_array.max())
    mask_surface = pygame.Surface(resolution, pygame.SRCALPHA)
    mask_pixels = pygame.surfarray.pixels_alpha(mask_surface)
    mask_pixels[:, :] = alpha_array.T
    del mask_pixels

    # Setze RGB auf weiß
    rgb = pygame.surfarray.pixels3d(mask_surface)
    rgb[:, :, :] = 255
    del rgb
    return mask_surface

def build_tilemap(mask_path, size, rows=9, cols=16, threshold=128):
    opacity_map = np.array(Image.open(mask_path).convert("L").resize(size))
    cell_h = size[1] // rows
    cell_w = size[0] // cols
    kacheln = []
    for row in range(rows):
        for col in range(cols):
            y1 = row * cell_h
            y2 = (row+1) * cell_h
            x1 = col * cell_w
            x2 = (col+1) * cell_w
            if np.max(opacity_map[y1:y2, x1:x2]) > threshold:
                # Jede Kachel weiß: ihre Grid-Position und ihr Frequenzband (Zeile)
                kacheln.append({
                    "x": x1, "y": y1, "w": cell_w, "h": cell_h, "row": row, "col": col
                })
    return kacheln

def compute_bands(audio, sample_rate, bands_n=9, band_noise_floor=None, normalize=True):
    # Berechne FFT
    fft = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    band_edges = np.logspace(np.log10(40), np.log10(sample_rate/2), bands_n+1)
    band_values = []
    for i in range(bands_n):
        idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i+1]))[0]
        if len(idx) > 0:
            band_values.append(np.sqrt(np.mean(fft[idx]**2)))
        else:
            band_values.append(0.0)
    band_values = np.array(band_values)

    # Dynamischer Noise-Floor-Abzug
    if band_noise_floor is not None:
        band_values = np.maximum(band_values - band_noise_floor, 0)

    # Adaptive Normalisierung: nur bei ausreichendem Signal
    if normalize:
        signal_sum = band_values.sum()
        if signal_sum > 0.05:
            band_values = band_values / (band_values.max() + 1e-8)

    if normalize:
        band_values = band_values * band_gains[:bands_n]
    band_values = np.clip(band_values, 0, 1)

    if normalize:
        band_values[band_values < 0.07] = 0

    return band_values

def main():
    global offset
    offset = load_offset()
    band_noise_floor = load_band_noise_floor(bands_n=9)
    if offset is None:
        print(f"Offset-Datei '{OFFSET_FILE}' fehlt! Bitte zuerst kalibrieren:\n   python3 {sys.argv[0]} calibrate\nProgramm beendet sich.")
        # Anzeige auf HDMI:
        pygame.init()
        info = pygame.display.Info()
        size = (info.current_w, info.current_h)
        win = pygame.display.set_mode(size, pygame.FULLSCREEN)
        pygame.display.set_caption("MusicToLight Pegelanzeige")
        pygame.mouse.set_visible(False)
        font = pygame.font.SysFont("monospace", 80)
        win.fill((0, 0, 0))
        msg = font.render("Offset fehlt!", True, (255, 0, 0))
        win.blit(msg, (size[0]//2 - msg.get_width()//2, size[1]//2 - msg.get_height()//2))
        pygame.display.flip()
        time.sleep(4)
        pygame.quit()
        return

    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()

    pygame.init()
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    win = pygame.display.set_mode(size, pygame.FULLSCREEN)
    pygame.display.set_caption("MusicToLight Pegelanzeige")
    pygame.mouse.set_visible(False)

    # --- Opacity-Map laden ---
    mask_path = os.path.join(os.path.dirname(__file__), OPACITY_MAP_FILE)
    # Opacity-Map und Kachelmap laden
    if not os.path.isfile(mask_path):
        alpha_mask = None
        kacheln = []
        print("Opacity-Map nicht gefunden:", mask_path)
    else:
        alpha_mask = load_mask_surface(mask_path, size)
        print("Opacity-Map geladen:", mask_path)
        kacheln = build_tilemap(mask_path, size, rows=9, cols=16)
        rows = 9
        cols = 16
        scroll_values = np.zeros((rows, cols), dtype=float)
        print(f"{len(kacheln)} von 144 Kacheln sichtbar auf dieser Map.")
        if mapping and alpha_mask:
            print("Zeige Testbild mit Opacity-Map... (beenden mit ESC oder ALT+F4)")
            running = True
            clock = pygame.time.Clock()
            test_surf = pygame.Surface(size, pygame.SRCALPHA)
            test_surf.fill((255, 255, 255, 255))  # Alles weiß
            test_surf.blit(alpha_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                win.fill((0, 0, 0))
                win.blit(test_surf, (0, 0))
                pygame.display.flip()
                clock.tick(30)
            pygame.quit()
            return  # danach nichts mehr machen!

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Lautstärke holen (max links/rechts – oder du nimmst Mittelwert) und Farbe je nach Pegel
        loud = min(max(audio_levels[0] - offset[0], audio_levels[1] - offset[1]), 1.0)
        loud = max(loud, 0.0)
        loud_vis = min(loud * boost, 1.0)
        color = (int(loud_vis * 255), 0, int((1 - loud_vis) * 128) + 64)

        # 1. Hol dir den aktuellen Audio-Buffer (links+rechts)
        audio_mix = (audio_levels[0] + audio_levels[1]) / 2

        # 2. Du brauchst das aktuelle Rohsignal für die FFT, nicht nur den Pegel.
        # Tipp: Sammle z. B. in audio_callback() die letzten 1024 Samples:
        # global audio_buffer = np.zeros(BUFFER_SIZE)
        # ... im Callback:
        # audio_buffer[:] = (left + right) / 2

        # Im Renderloop:
        band_values = compute_bands(audio_buffer, SAMPLE_RATE, bands_n=9, band_noise_floor=band_noise_floor)
        # Sound "läuft" von links nach rechts:
        for row in range(rows):
            for col in reversed(range(1, cols)):
                scroll_values[row, col] = scroll_values[row, col - 1]
            # Linkeste Kachel immer aktuell!
            bands_n = len(band_values)
            scroll_values[row, 0] = band_values[bands_n - 1 - row]

        # 3. Visualisiere:
        vis_surface = pygame.Surface(size, pygame.SRCALPHA)
        for k in kacheln:
            row = k["row"]
            col = k["col"]
            val = scroll_values[row, col]
            scale = 0.1 + val * 0.85
            w = int(k["w"] * scale)
            h = int(k["h"] * scale)
            x = k["x"] + (k["w"] - w) // 2
            y = k["y"] + (k["h"] - h) // 2
            color = get_band_color(row, rows)
            pygame.draw.rect(vis_surface, color, (x, y, w, h))

        # *** Am Ende KEIN vis_surface.blit(alpha_mask, ...) mehr! ***
        win.fill((0, 0, 0))
        win.blit(vis_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate()
    elif len(sys.argv) > 1 and sys.argv[1] == "mapping":
        mapping = True
        main()
    else:
        main()