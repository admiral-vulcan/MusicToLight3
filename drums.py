import sounddevice as sd
import numpy as np
import threading
import time
import json
import os
import sys
from scipy.signal import butter, lfilter

# Performance setting
FPS = 60  # Default is 60 fps, change if screens or Pi are performing poorly

# Audio settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 2
flip_channels = True
BAND_HISTORY_LEN = 5 * FPS  # seconds times FPS
OFFSET_CORRECTION = 0.015
SELECTED_CHANNEL = 'left'

# Buffers and gains
audio_levels = np.zeros(2)
audio_buffer = np.zeros((BUFFER_SIZE * 4, 2))  # Stereo

# Audio threading
audio_running = threading.Event()

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

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def band_energy(audio, fs, lowcut, highcut):
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered = lfilter(b, a, audio)
    return np.sqrt(np.mean(filtered**2))

def zero_crossings(audio):
    return ((audio[:-1] * audio[1:]) < 0).sum() / len(audio)

def spectral_flux(mag_prev, mag_cur):
    return np.sum((mag_cur - mag_prev).clip(0, None))

def find_next_filename(prefix):
    """Find the next available filename with 3-digit numbering."""
    i = 0
    while True:
        filename = f"{prefix}{i:03d}.json"
        if not os.path.exists(filename):
            return filename
        i += 1

def drum_analysis(audio, samplerate, state, json_path):
    """
    Extended adaptive drum analysis with background calibration and JSON-logging.
    """
    INIT_FRAMES = 60
    threshold_factor = 1.5

    if 'mag_prev' not in state:
        state['mag_prev'] = None

    if 'frame_count' not in state:
        state.update({
            'frame_count': 0,
            'peak_max': 0,
            'rms_max': 0,
            'sharpness_max': 0,
            'flux_max': 0,
            'zcr_max': 0,
            'bass_max': 0,
            'mid_max': 0,
            'high_max': 0,
            'bass_attack_max': 0,
            'rms_deriv_max': 0,
        })

    state['frame_count'] += 1

    # --- Feature Calculation ---
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    sharpness = peak / (rms + 1e-8)

    mag = np.abs(np.fft.rfft(audio))
    flux = 0
    if state['mag_prev'] is not None:
        flux = spectral_flux(state['mag_prev'], mag)
    state['mag_prev'] = mag

    bass = band_energy(audio, samplerate, 20, 120)
    mid = band_energy(audio, samplerate, 120, 2000)
    high = band_energy(audio, samplerate, 2000, 8000)

    band_sum = bass + mid + high + 1e-12
    rel_bass = bass / band_sum
    rel_mid = mid / band_sum
    rel_high = high / band_sum

    N = len(audio)
    split = int(0.05 * samplerate)
    bass_attack = 0
    if N > split:
        bass_first = band_energy(audio[:split], samplerate, 20, 120)
        bass_rest = band_energy(audio[split:], samplerate, 20, 120)
        bass_attack = (bass_first - bass_rest) / (bass_rest + 1e-8)

    zcr = zero_crossings(audio)

    if 'rms_prev' not in state:
        state['rms_prev'] = rms
    rms_deriv = rms - state['rms_prev']
    state['rms_prev'] = rms

    # --- Calibration Phase ---
    for key, val in [
        ('peak_max', peak), ('rms_max', rms), ('sharpness_max', sharpness),
        ('flux_max', flux), ('zcr_max', zcr),
        ('bass_max', bass), ('mid_max', mid), ('high_max', high),
        ('bass_attack_max', bass_attack), ('rms_deriv_max', rms_deriv)
    ]:
        state[key] = max(state[key], val)

    if state['frame_count'] == INIT_FRAMES:
        print("Ready for analysis! Thresholds calibrated.")
        for k in [k for k in state.keys() if k.endswith('_max')]:
            print(f"{k}: {state[k]:.5f}")
        print("-" * 60)
        return None

    if state['frame_count'] <= INIT_FRAMES:
        return None

    # --- Event Detection ---
    is_event = any([
        peak > state['peak_max'] * threshold_factor,
        rms > state['rms_max'] * threshold_factor,
        sharpness > state['sharpness_max'] * threshold_factor,
        flux > state['flux_max'] * threshold_factor,
        zcr > state['zcr_max'] * threshold_factor,
        bass > state['bass_max'] * threshold_factor,
        mid > state['mid_max'] * threshold_factor,
        high > state['high_max'] * threshold_factor,
        bass_attack > state['bass_attack_max'] * threshold_factor,
        rms_deriv > state['rms_deriv_max'] * threshold_factor,
    ])

    if is_event:
        print("-" * 60)
        print("EXTENDED adaptive_drum_analysis()")
        print(f"Peak:          {peak:.5f} (Thresh: {state['peak_max']:.5f})")
        print(f"RMS:           {rms:.5f} (Thresh: {state['rms_max']:.5f})")
        print(f"Sharpness:     {sharpness:.2f} (Thresh: {state['sharpness_max']:.2f})")
        print(f"Spectral Flux: {flux:.5f} (Thresh: {state['flux_max']:.5f})")
        print(f"Bass:          {bass:.5f} (Thresh: {state['bass_max']:.5f})")
        print(f"Mid:           {mid:.5f} (Thresh: {state['mid_max']:.5f})")
        print(f"High:          {high:.5f} (Thresh: {state['high_max']:.5f})")
        print(f"Bass Attack:   {bass_attack:.2f} (Thresh: {state['bass_attack_max']:.2f})")
        print(f"ZeroCrossRate: {zcr:.4f} (Thresh: {state['zcr_max']:.4f})")
        print(f"RMS_Deriv:     {rms_deriv:.5f} (Thresh: {state['rms_deriv_max']:.5f})")
        print("-" * 60)

    # --- Data Collection ---
    event_dict = {
        'frame': state['frame_count'],
        'peak': float(peak),
        'rms': float(rms),
        'sharpness': float(sharpness),
        'spectral_flux': float(flux),
        'bass': float(bass),
        'mid': float(mid),
        'high': float(high),
        'rel_bass': float(rel_bass),
        'rel_mid': float(rel_mid),
        'rel_high': float(rel_high),
        'bass_attack': float(bass_attack),
        'zcr': float(zcr),
        'rms_deriv': float(rms_deriv),
        'is_event': bool(is_event)
    }

    # --- Robust JSON Logging (one line per frame) ---
    try:
        with open(json_path, "a") as f:
            f.write(json.dumps(event_dict) + "\n")
    except Exception as e:
        print("Fehler beim Schreiben der JSON:", e)

    return event_dict

def main():
    if len(sys.argv) < 2:
        print("Aufruf: python drums.py <basename>")
        print("Beispiel: python drums.py kick")
        sys.exit(1)

    base = sys.argv[1]
    json_path = find_next_filename(base)

    print(f"JSON-Datei: {json_path}")

    state = {}

    # --- Audio-Thread starten ---
    audio_running.set()
    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_audio.start()

    frame_len = BUFFER_SIZE  # Deine gewählte Bufferlänge (passt zu FPS)

    try:
        while True:
            # --- Aktuellen Bufferbereich für Analyse holen ---
            # Nimm den gewünschten Kanal
            if SELECTED_CHANNEL == 'left':
                audio = audio_buffer[-frame_len:, 0].copy()
            elif SELECTED_CHANNEL == 'right':
                audio = audio_buffer[-frame_len:, 1].copy()
            else:
                audio = np.mean(audio_buffer[-frame_len:, :], axis=1)  # beide Kanäle

            if np.all(audio == 0):
                continue  # Skip until buffer is filled
            drum_analysis(audio, SAMPLE_RATE, state, json_path)

            # Zeitsteuerung – damit ca. FPS eingehalten werden!
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        audio_running.clear()
        print("\nProgramm beendet (STRG+C). JSON-Datei ist vollständig.")

if __name__ == "__main__":
    main()