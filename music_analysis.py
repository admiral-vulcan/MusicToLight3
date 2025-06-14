# music_analysis.py

import numpy as np
from collections import deque

# Global memory for rolling Techno detection
# Use one buffer per song/context, if needed!
_TECHNO_HISTORY = deque(maxlen=30)  # Last 30 results (~1 sec at 30 FPS)

def normalize_audio(audio):
    """
    Normalizes the audio array to [-1, 1] if necessary.
    """
    if np.issubdtype(audio.dtype, np.floating):
        maxval = np.max(np.abs(audio))
        if maxval <= 1.05:
            return audio.copy()
        return audio / (maxval + 1e-8)
    elif np.issubdtype(audio.dtype, np.integer):
        return audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio / (np.max(np.abs(audio)) + 1e-8)

def is_techno(audio, samplerate, debug_level=0, FPS=30, mean_secs=1.0, history_length=30):
    """
    Classifies the input as 'Hard Techno' or 'Chill'.
    Keeps a rolling history of recent results to smooth detection.
    Returns True if >49% of recent detections are positive, else False.
    Also prints a rolling Techno rating if debug_level>=1.

    - audio: 1D numpy array, mono.
    - samplerate: sample rate in Hz.
    - debug_level: 0 = silent, 1 = summary, 2 = full debug.
    - FPS: Detection calls per second (used for history smoothing).
    - mean_secs: Rolling window for stats (not history).
    - history_length: Number of past detections to remember (default 30).

    You can call is_techno() as before.
    """
    global _TECHNO_HISTORY

    # --- Thresholds for scoring (easy to tune) ---
    LOUDNESS_THRESH      = -26    # dBFS
    REL_BASS_THRESH      = 0.15
    REL_HI_THRESH        = 0.005
    MEAN_FLUX_THRESH     = 8
    ZCR_THRESH           = 0.020
    REL_MID_THRESH       = 0.13

    HIHAT_MUST_HAVE      = 0.01

    # Bass Techno Path thresholds
    REL_BASS_BASS_TECHNO = 0.33
    REL_MID_BASS_TECHNO  = 0.10
    MEAN_FLUX_BASS_TECHNO= 10
    
    # --- Normalize audio to [-1, 1] ---
    audio = normalize_audio(audio)
    N = len(audio)
    if debug_level >= 2:
        print(f"After normalization: max={np.max(audio):.3f}, min={np.min(audio):.3f}, mean={np.mean(audio):.3f}, N={N}")

    # --- Feature extraction ---
    rms = np.sqrt(np.mean(audio ** 2))
    loudness = 20 * np.log10(rms + 1e-8)

    windowed = audio * np.hanning(N)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(N, 1 / samplerate)
    energy = spectrum ** 2

    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    bass_band = band(35, 120)
    mid_band = band(300, 3000)
    hi_band = band(6000, 18000)

    bass_energy = np.sum(energy[bass_band])
    mid_energy = np.sum(energy[mid_band])
    hi_energy = np.sum(energy[hi_band])
    total_energy = np.sum(energy) + 1e-8

    rel_bass = bass_energy / total_energy
    rel_mid = mid_energy / total_energy
    rel_hi = hi_energy / total_energy

    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * N)

    # Spectral flux (transient activity)
    hop = N // 4
    flux = 0.0
    prev = np.abs(np.fft.rfft(audio[:hop]))
    for i in range(hop, N, hop):
        current = np.abs(np.fft.rfft(audio[i:i+hop]))
        flux += np.sum((current - prev).clip(0))
        prev = current
    mean_flux = flux / (N / hop)

    # --- Scoring (tuned for normalized audio) ---
    score = 0
    if loudness > LOUDNESS_THRESH:      score += 1
    if rel_bass > REL_BASS_THRESH:      score += 1
    if rel_hi > REL_HI_THRESH:          score += 1
    if mean_flux > MEAN_FLUX_THRESH:    score += 1
    if zcr > ZCR_THRESH:                score += 1
    if rel_mid < REL_MID_THRESH:        score += 1

    # Path 1: "Classic Techno" (some hi-hat presence and overall high score)
    classic_techno = (score >= 3) and (rel_hi > HIHAT_MUST_HAVE)

    # Path 2: "Bass Techno" (very bass-heavy, minimal mids, high movement, allows missing hi-hats)
    bass_techno = (rel_bass > REL_BASS_BASS_TECHNO) and (rel_mid < REL_MID_BASS_TECHNO) and (mean_flux > MEAN_FLUX_BASS_TECHNO)

    # Final frame decision: either path is sufficient
    frame_is_techno = classic_techno or bass_techno

    # --- History management (rolling voting) ---
    if len(_TECHNO_HISTORY) != history_length:
        _TECHNO_HISTORY = deque(_TECHNO_HISTORY, maxlen=history_length)
    _TECHNO_HISTORY.append(frame_is_techno)

    percent_techno = 100 * np.mean(_TECHNO_HISTORY) if _TECHNO_HISTORY else 0.0
    is_really_techno = percent_techno > 39

    # --- Debug output ---
    if debug_level >= 1:
        print(f"\nTechno-Detection DEBUG (Rolling):")
        print(f" loudness      = {loudness:.2f} dBFS   (> {LOUDNESS_THRESH}: loud enough)")
        print(f" rel_bass      = {rel_bass:.3f}        (> {REL_BASS_THRESH}: much bass)")
        print(f" rel_mid       = {rel_mid:.3f}         (< {REL_MID_THRESH}: not mid-heavy)")
        print(f" rel_hi        = {rel_hi:.3f}          (> {REL_HI_THRESH}: some hi-hat)")
        print(f" mean_flux     = {mean_flux:.1f}       (> {MEAN_FLUX_THRESH}: transient)")
        print(f" zcr           = {zcr:.3f}             (> {ZCR_THRESH}: some zero crossings)")
        print(f" --> SCORE     = {score}/6 (frame: {'HARD TECHNO' if frame_is_techno else 'not techno'})")
        print(f"   Classic Techno Path: {'YES' if classic_techno else 'no'},  Bass Techno Path: {'YES' if bass_techno else 'no'}")
        print(f" Rolling Techno rating: {percent_techno:.1f}% (last {history_length} frames)")
        print(f" ==> FINAL DECISION: {'HARD TECHNO' if is_really_techno else 'not techno'}")
    if debug_level == 2:
        print(f"   raw: bass={bass_energy:.2e}, mid={mid_energy:.2e}, hi={hi_energy:.2e}, total={total_energy:.2e}")

    return is_really_techno



def analyze_energy(audio, sample_rate, FPS=60, mean_secs=5.0):
    """
    Returns dict with absolute and mean energy for low, mid, high, and overall frequency bands.
    - "absolute": raw RMS of current frame per band.
    - "mean": rolling mean over last mean_secs (using FPS).
    - Keeps its own state (rolling history) between calls.
    """

    # --- Frequency band edges ---
    LOW_EDGE = 40
    LOW_HIGH = 250
    MID_HIGH = 4000
    HIGH_EDGE = sample_rate // 2

    # --- FFT windowed ---
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)

    # --- Band RMS calculations ---
    def band_energy(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return np.sqrt(np.mean(spectrum[idx] ** 2)) if np.any(idx) else 0.0

    abs_low  = band_energy(LOW_EDGE, LOW_HIGH)
    abs_mid  = band_energy(LOW_HIGH, MID_HIGH)
    abs_high = band_energy(MID_HIGH, HIGH_EDGE)
    abs_total = np.sqrt(np.mean(spectrum ** 2))

    # --- Rolling mean state ---
    n_frames = int(mean_secs * FPS)
    if not hasattr(analyze_energy, "history"):
        # Per-band history buffer (initialized with zeros)
        analyze_energy.history = {
            "low": deque([0.0]*n_frames, maxlen=n_frames),
            "mid": deque([0.0]*n_frames, maxlen=n_frames),
            "high": deque([0.0]*n_frames, maxlen=n_frames),
            "overall": deque([0.0]*n_frames, maxlen=n_frames)
        }
        analyze_energy.n_frames = n_frames
    # Resize history buffer if parameters change
    if n_frames != analyze_energy.n_frames:
        for key in analyze_energy.history:
            last = list(analyze_energy.history[key])
            analyze_energy.history[key] = deque(
                (last + [0.0]*n_frames)[:n_frames], maxlen=n_frames)
        analyze_energy.n_frames = n_frames

    # --- Append current values ---
    analyze_energy.history["low"].append(abs_low)
    analyze_energy.history["mid"].append(abs_mid)
    analyze_energy.history["high"].append(abs_high)
    analyze_energy.history["overall"].append(abs_total)

    # --- Compute rolling means ---
    mean_low  = np.mean(analyze_energy.history["low"])
    mean_mid  = np.mean(analyze_energy.history["mid"])
    mean_high = np.mean(analyze_energy.history["high"])
    mean_total= np.mean(analyze_energy.history["overall"])

    return {
        "absolute": {
            "low": abs_low,
            "mid": abs_mid,
            "high": abs_high,
            "overall": abs_total
        },
        "mean": {
            "low": mean_low,
            "mid": mean_mid,
            "high": mean_high,
            "overall": mean_total
        }
    }

def detect_drums(audio, samplerate, state=None, debug_level=0, FPS=60, mean_secs=5.0):
    """
    Robust per-frame drum detection for EDM/Techno/Pop. Detects KickSub, KickPunch, Snare/Clap, and HiHat.
    Uses spectral centroid, absolute & relative band energy, temporal (attack) and statistical features.
    Debug output:
        0 = no prints,
        1 = print only when something is detected,
        2 = always print all features.
    """
    if state is None:
        state = {}

    # --- Compute per-band energy info (RMS, means, etc.) ---
    energy_info = analyze_energy(audio, samplerate, FPS=FPS, mean_secs=mean_secs)

    # --- Ratio: instantaneous energy vs. rolling mean (per band) ---
    kick_energy_ratio   = energy_info["absolute"]["low"]  / (energy_info["mean"]["low"]  + 1e-8)
    snare_energy_ratio  = energy_info["absolute"]["mid"]  / (energy_info["mean"]["mid"]  + 1e-8)
    hihat_energy_ratio  = energy_info["absolute"]["high"] / (energy_info["mean"]["high"] + 1e-8)

    # --- FFT & spectral features ---
    windowed = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1 / samplerate)
    energy = spectrum ** 2

    # Spectral centroid: frequency "center of mass" (Hz)
    spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)

    # --- Frequency band selectors (for attack & debug) ---
    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    kick_band   = band(43, 90)
    sub_band    = band(25, 45)
    snare_band  = band(145, 320)
    clap_band   = band(2000, 5000)
    hihat_band  = band(8800, 13000)
    mid_band    = band(300, 2500)
    hi_band     = band(7000, 16000)

    # --- Instantaneous band energies (for attack/transient detection) ---
    bass_now   = np.sum(energy[kick_band]) + np.sum(energy[sub_band])
    snare_now  = np.sum(energy[snare_band]) + np.sum(energy[clap_band])
    hihat_now  = np.sum(energy[hihat_band]) + np.sum(energy[hi_band])
    mid_now    = np.sum(energy[mid_band])
    total_now  = np.sum(energy) + 1e-8  # Avoid division by zero

    rel_bass   = bass_now / total_now
    rel_snare  = snare_now / total_now
    rel_hihat  = hihat_now / total_now
    rel_mid    = mid_now / total_now

    # --- Temporal/statistical features ---
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))  # Zero-crossing rate per frame
    rms = np.sqrt(np.mean(audio ** 2))                               # Root Mean Square: loudness
    sharpness = np.max(np.abs(audio)) / (rms + 1e-8)                 # Peak/RMS: transient ratio

    # --- Feature history for "attack" computation (short-term vs long-term) ---
    N_HISTORY = 4
    if 'mag_prev' not in state:
        state['mag_prev'] = np.zeros_like(spectrum)
    spectral_flux = np.sum((spectrum - state['mag_prev']).clip(0))
    state['mag_prev'] = spectrum.copy()
    for key in ['bass_hist', 'snare_hist', 'hihat_hist']:
        if key not in state:
            state[key] = [0.0] * N_HISTORY

    bass_attack   = bass_now  - np.mean(state['bass_hist'])
    snare_attack  = snare_now - np.mean(state['snare_hist'])
    hihat_attack  = hihat_now - np.mean(state['hihat_hist'])

    state['bass_hist'].pop(0)
    state['bass_hist'].append(bass_now)
    state['snare_hist'].pop(0)
    state['snare_hist'].append(snare_now)
    state['hihat_hist'].pop(0)
    state['hihat_hist'].append(hihat_now)

    # --- Drum detection thresholds (well-documented for tuning) ---
    kick_sub_thresh = {
        'rel_bass': 0.2,
        'bass_attack': 0.01,
        'sharpness': 1.2,
        'spectral_flux': 200,  # vorher: 40 !
        'centroid_max': 700,  # vorher: 400
        'energy_ratio': 2.0
    }
    kick_punch_thresh = {
        'rel_bass': 0.1,
        'bass_attack': 0.002,
        'sharpness': 2.5,
        'spectral_flux': 100,  # vorher: 30
        'centroid_max': 2000,  # vorher: 700
        'energy_ratio': 1.2  # vorher: 1.5 (macht’s noch sensibler)
    }
    snare_thresh = {
        'rel_snare': 0.004,  # vorher: 0.005
        'snare_attack': 0.0005,  # vorher: 0.0007
        'zcr_lo': 0.04,  # vorher: 0.15
        'zcr_hi': 0.2,  # vorher: 0.31
        'spectral_flux': 0.0002,  # vorher: 0.0005
        'sharpness': 0.8,  # vorher: 1.0
        'centroid_lo': 200,  # vorher: 400
        'centroid_hi': 5000,  # vorher: 4000
        'energy_ratio': 1.1  # vorher: 1.3
    }
    hihat_thresh = {
        'rel_hihat': 0.008,  # vorher: 0.01
        'hihat_attack': 0.003,  # vorher: 0.005
        'zcr': 0.03,  # vorher: 0.09
        'sharpness': 2.0,  # vorher: 3.0
        'spectral_flux': 1.5,  # vorher: 2.0
        'centroid_min': 2500,  # vorher: 3500
        'energy_ratio': 1.05  # vorher: 1.1
    }

    # --- Detection logic: all conditions must be fulfilled for a positive ---
    detected = []

    kicksub_score = 0
    if rel_bass > kick_sub_thresh['rel_bass']:
        kicksub_score += 1
    if bass_attack > kick_sub_thresh['bass_attack']:
        kicksub_score += 1
    if sharpness > kick_sub_thresh['sharpness']:
        kicksub_score += 1
    if spectral_flux < kick_sub_thresh['spectral_flux']:
        kicksub_score += 1
    if spectral_centroid < kick_sub_thresh['centroid_max']:
        kicksub_score += 1
    if kick_energy_ratio > kick_sub_thresh['energy_ratio']:
        kicksub_score += 1

    if kicksub_score >= 5:  # oder sogar >= 4 (du kannst spielen)
        detected.append("KickSub")

    # --- KickPunch (Score statt reines AND) ---
    kickpunch_score = 0
    if rel_bass > kick_punch_thresh['rel_bass']:
        kickpunch_score += 1
    if bass_attack > kick_punch_thresh['bass_attack']:
        kickpunch_score += 1
    if sharpness > kick_punch_thresh['sharpness']:
        kickpunch_score += 1
    if spectral_flux > kick_punch_thresh['spectral_flux']:
        kickpunch_score += 1
    if spectral_centroid < kick_punch_thresh['centroid_max']:
        kickpunch_score += 1
    if kick_energy_ratio > kick_punch_thresh['energy_ratio']:
        kickpunch_score += 1

    if kickpunch_score >= 5:
        detected.append("KickPunch")

    if (
        rel_snare > snare_thresh['rel_snare'] and
        snare_attack > snare_thresh['snare_attack'] and
        zcr > snare_thresh['zcr_lo'] and zcr < snare_thresh['zcr_hi'] and
        spectral_flux > snare_thresh['spectral_flux'] and
        sharpness > snare_thresh['sharpness'] and
        snare_thresh['centroid_lo'] < spectral_centroid < snare_thresh['centroid_hi'] and
        snare_energy_ratio > snare_thresh['energy_ratio']
    ):
        detected.append("Snare/Clap")

    if (
        rel_hihat > hihat_thresh['rel_hihat'] and
        # hihat_attack > hihat_thresh['hihat_attack'] and  # Enable for strict detection
        zcr > hihat_thresh['zcr'] and
        # sharpness > hihat_thresh['sharpness'] and         # Enable for even stricter gating
        spectral_flux > hihat_thresh['spectral_flux'] and
        spectral_centroid > hihat_thresh['centroid_min'] and
        hihat_energy_ratio > hihat_thresh['energy_ratio']
    ):
        detected.append("HiHat")

    # --- Debug output selection ---
    debug_should_print = (
        (debug_level == 2) or
        (debug_level == 1 and detected)
    )

    if debug_should_print:
        print("\nDrum-Detection DEBUG:")
        print(f" spectral_centroid = {spectral_centroid:.1f} Hz (KickSub < {kick_sub_thresh['centroid_max']} | "
              f"KickPunch < {kick_punch_thresh['centroid_max']} | Snare {snare_thresh['centroid_lo']}–{snare_thresh['centroid_hi']} | HiHat > {hihat_thresh['centroid_min']})")
        print(f" rel_bass   = {rel_bass:.5f}   (KickSub > {kick_sub_thresh['rel_bass']} | KickPunch > {kick_punch_thresh['rel_bass']})")
        print(f" bass_attack= {bass_attack:.5f} (KickSub > {kick_sub_thresh['bass_attack']} | KickPunch > {kick_punch_thresh['bass_attack']})")
        print(f" sharpness  = {sharpness:.5f}  (KickSub > {kick_sub_thresh['sharpness']} | KickPunch > {kick_punch_thresh['sharpness']})")
        print(f" spectral_flux = {spectral_flux:.5f} (KickSub < {kick_sub_thresh['spectral_flux']} | KickPunch > {kick_punch_thresh['spectral_flux']})")
        print(f" kick_energy_ratio = {kick_energy_ratio:.2f} (KickSub > {kick_sub_thresh['energy_ratio']} | KickPunch > {kick_punch_thresh['energy_ratio']})")
        print(f" rel_snare  = {rel_snare:.5f}  (Snare > {snare_thresh['rel_snare']})")
        print(f" snare_attack = {snare_attack:.5f} (Snare > {snare_thresh['snare_attack']})")
        print(f" zcr        = {zcr:.5f}        (Snare {snare_thresh['zcr_lo']}...{snare_thresh['zcr_hi']})")
        print(f" spectral_flux = {spectral_flux:.5f} (Snare > {snare_thresh['spectral_flux']})")
        print(f" sharpness  = {sharpness:.5f}  (Snare > {snare_thresh['sharpness']})")
        print(f" snare_energy_ratio = {snare_energy_ratio:.2f} (Snare > {snare_thresh['energy_ratio']})")
        print(f" rel_hihat  = {rel_hihat:.5f}  (HiHat > {hihat_thresh['rel_hihat']})")
        print(f" hihat_attack = {hihat_attack:.5f} (HiHat > {hihat_thresh['hihat_attack']})")
        print(f" zcr        = {zcr:.5f}        (HiHat > {hihat_thresh['zcr']})")
        print(f" sharpness  = {sharpness:.5f}  (HiHat > {hihat_thresh['sharpness']})")
        print(f" spectral_flux = {spectral_flux:.5f} (HiHat > {hihat_thresh['spectral_flux']})")
        print(f" hihat_energy_ratio = {hihat_energy_ratio:.2f} (HiHat > {hihat_thresh['energy_ratio']})")
        print(f" --> DETECTED: {detected}")

    return detected

