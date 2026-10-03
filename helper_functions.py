import numpy as np
import scipy.signal as signal
import json
from pathlib import Path

def simulate_skin_impedance_raw(raw_emg, noise_std):
    """
    Simulate the skin impedance and sweat by adding WGN
    
    Parameters

    raw_emg: np.ndarray of shape (n_samples, n_channels) 
    noise_std:  standard deviation of the injected white noise
    """
    if raw_emg.ndim != 2:
        raise ValueError("Expected raw_emg with shape (n_samples, n_channels).")

    noisy_emg = raw_emg.copy()
    noise = np.random.normal(loc=0.0, scale=noise_std, size=noisy_emg.shape)

    noisy_emg = raw_emg + noise

    return noisy_emg

def simulate_fatigue_raw(raw_emg, fs, fatigue_end_factor=0.6, lp_cutoff=150.0):
    """
    Simulate the muscle fatigue by shifting the median frequency using the low-pass filtering of a raw signal
    
    Parameters

    raw_emg: np.ndarray of shape (n_samples, n_channels) 
    fs:  sampling frequency
    fatigue_end_factor: final amplitude at end of rep (e.g. 0.6 = 60%)
    lp_cutoff: low-pass cutoff to mimic median frequency shift
    """
    if raw_emg.ndim != 2 or raw_emg.shape[0] < 5:
        return raw_emg

    T, C = raw_emg.shape

    # Progressive amplitude drop
    attenuation = np.linspace(1.0, fatigue_end_factor, T).reshape(-1, 1)
    emg_att = raw_emg * attenuation

    # Design low-pass filter (Butterworth)
    nyq = fs / 2.0
    wn = lp_cutoff / nyq
    b, a = signal.butter(4, wn, btype='low')

    # Apply along time for each channel
    emg_lp = signal.filtfilt(b, a, emg_att, axis=0)

    return emg_lp

def simulate_electrode_shift_raw(raw_emg, shift_strength=0.3):
    """
    Simulate electrode shift by spatially mixing neighbouring sEMG channels.

    Parameters
  
    raw_emg : np.ndarray of shape (n_samples, n_channels)
    shift_strength : how strongly each channel is mixed with its neighbours (0=no shift, 1=only neighbours).
    """
    if raw_emg.ndim != 2:
        raise ValueError("Expected raw_emg with shape (n_samples, n_channels).")

    n_samples, n_channels = raw_emg.shape
    mixed = raw_emg.copy()

    for ch in range(n_channels):
        neighbours = []
        if ch > 0:
            neighbours.append(raw_emg[:, ch - 1])
        if ch < n_channels - 1:
            neighbours.append(raw_emg[:, ch + 1])

        if neighbours:
            neighbour_avg = np.mean(neighbours, axis=0)
            mixed[:, ch] = (1.0 - shift_strength) * raw_emg[:, ch] + shift_strength * neighbour_avg  # weighted average

    return mixed


def simulate_sensor_failure_raw(raw_emg, failed_channels, noise_std=1e-4):
    """
    Simulate catastrophic sensor failure by replacing selected channels
    with very low-amplitude white noise, mimicking open-circuit hiss.

    Parameters
   
    raw_emg : np.ndarray of shape (n_samples, n_channels)
    failed_channels : zero-based indices of channels that have failed.
    noise_std : standard deviation of the injected white noise.
    """
    if raw_emg.ndim != 2:
        raise ValueError("Expected raw_emg with shape (n_samples, n_channels).")

    n_samples, n_channels = raw_emg.shape
    failed_channels = [ch for ch in failed_channels if 0 <= ch < n_channels]

    if not failed_channels:
        return raw_emg

    failed_emg = raw_emg.copy()
    noise = np.random.normal(loc=0.0, scale=noise_std, size=(n_samples, len(failed_channels)))

    for i, ch in enumerate(failed_channels):
        failed_emg[:, ch] = noise[:, i]

    return failed_emg


def save_best_hparams_json(path: str, payload: dict) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_best_hparams_json(path: str) -> tuple[dict, dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "hparams" not in payload:
        raise ValueError("Best-params file must be a JSON object containing a top-level 'hparams' key.")
    h = payload["hparams"]
    if not isinstance(h, dict):
        raise ValueError("'hparams' must be a JSON object.")
    return h, payload