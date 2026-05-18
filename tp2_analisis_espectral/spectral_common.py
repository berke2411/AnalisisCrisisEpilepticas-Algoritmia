"""
Funciones comunes para la Practica 2 - Analisis espectral EEG.

Librerias usadas:
pip install numpy scipy matplotlib pyedflib
"""

import numpy as np
from pathlib import Path
from scipy import signal
from pyedflib import highlevel


BASE_DIR = Path(__file__).resolve().parent

# Registro elegido para el analisis principal del TP.
# Tiempos extraidos de chb20-summary.txt.
RECORDS = {
    "chb20_12": {"file": "chb20_12.edf", "start_sec": 94, "end_sec": 123},
}
DEFAULT_RECORD = "chb20_12"


def find_edf_file(filename):
    candidates = [
        BASE_DIR.parent / "archivos" / filename,  # repo/archivos/chb20_xx.edf
        BASE_DIR / "archivos" / filename,         # tp2_analisis_espectral/archivos/chb20_xx.edf
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


EDF_FILE = str(find_edf_file(RECORDS[DEFAULT_RECORD]["file"]))
START_SEC = RECORDS[DEFAULT_RECORD]["start_sec"]
END_SEC = RECORDS[DEFAULT_RECORD]["end_sec"]
WINDOW_SEC = 120

BANDS = {
    "delta": (0, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 64),
}


def cargar_edf(path=EDF_FILE):
    signals, signal_headers, header = highlevel.read_edf(path)
    fs = int(signal_headers[0].get("sample_frequency", signal_headers[0].get("sample_rate")))
    labels = [h.get("label", f"Canal {i}") for i, h in enumerate(signal_headers)]
    return signals, fs, labels, header


def extract_and_center(sig, start, end):
    segment = sig[:, start:end]
    return segment - segment.mean(axis=1, keepdims=True)


def preparar_segmentos(signals, fs, start_sec=START_SEC, end_sec=END_SEC):
    start_sample = start_sec * fs
    end_sample = end_sec * fs
    window_samples = WINDOW_SEC * fs
    total_samples = signals.shape[1]

    inicio_before = max(0, start_sample - window_samples)
    fin_after = min(total_samples, end_sample + window_samples)

    before = extract_and_center(signals, inicio_before, start_sample)
    crisis = extract_and_center(signals, start_sample, end_sample)
    after = extract_and_center(signals, end_sample, fin_after)
    total = np.concatenate((before, crisis, after), axis=1)

    refs = {
        "start_sample": start_sample,
        "end_sample": end_sample,
        "inicio_before": inicio_before,
        "fin_after": fin_after,
        "crisis_ini_total_sec": before.shape[1] / fs,
        "crisis_fin_total_sec": (before.shape[1] + crisis.shape[1]) / fs,
    }
    return {"before": before, "crisis": crisis, "after": after, "total": total}, refs


def cargar_datos(record_key=DEFAULT_RECORD):
    cfg = RECORDS[record_key]
    edf_file = str(find_edf_file(cfg["file"]))
    signals, fs, labels, header = cargar_edf(edf_file)
    segmentos, refs = preparar_segmentos(signals, fs, cfg["start_sec"], cfg["end_sec"])
    return signals, fs, labels, header, segmentos, refs


def calcular_fft(seg, fs):
    n = seg.shape[1]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(seg, axis=1)) / n
    return freqs, fft_mag


def calcular_psd_welch(seg, fs, nperseg=None, overlap=0.5):
    if nperseg is None:
        nperseg = min(4 * fs, seg.shape[1])
    noverlap = int(nperseg * overlap)
    freqs, psd = signal.welch(
        seg,
        fs=fs,
        axis=1,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


def calcular_periodograma(seg, fs):
    freqs, pxx = signal.periodogram(seg, fs=fs, axis=1, scaling="density")
    return freqs, pxx


def calcular_stft(seg, fs, window_sec=2, overlap=0.50):
    nperseg = min(int(window_sec * fs), seg.shape[1])
    noverlap = int(nperseg * overlap)
    freqs, times, zxx = signal.stft(
        seg,
        fs=fs,
        axis=1,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )
    power = np.abs(zxx) ** 2
    return freqs, times, power


def calcular_espectrograma(seg, fs, window_sec, overlap):
    nperseg = min(int(window_sec * fs), seg.shape[1])
    noverlap = int(nperseg * overlap)
    freqs, times, spec = signal.spectrogram(
        seg,
        fs=fs,
        axis=1,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    return freqs, times, spec


def potencia_por_banda(freqs, psd):
    band_power = {}
    for name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        if np.any(mask):
            band_power[name] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        else:
            band_power[name] = np.zeros(psd.shape[0])
    return band_power


def potencia_bandas_tiempo(freqs, spec):
    band_power = {}
    for name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        if np.any(mask):
            band_power[name] = np.trapezoid(spec[:, mask, :], freqs[mask], axis=1)
        else:
            band_power[name] = np.zeros((spec.shape[0], spec.shape[2]))
    return band_power


def frecuencias_dominantes(freqs, spectrum, fmax=64, top_n=5):
    mask = (freqs > 0) & (freqs <= fmax)
    mean_spectrum = spectrum[:, mask].mean(axis=0)
    idx = np.argsort(mean_spectrum)[-top_n:][::-1]
    return list(zip(freqs[mask][idx], mean_spectrum[idx]))


def imprimir_resumen_carga(signals, fs, segmentos, record_key=DEFAULT_RECORD):
    cfg = RECORDS[record_key]
    print(f"Registro: {record_key}")
    print(f"Archivo EDF: {find_edf_file(cfg['file'])}")
    print(f"Crisis anotada: {cfg['start_sec']}s a {cfg['end_sec']}s")
    print(f"Frecuencia de muestreo: {fs} Hz")
    print(f"Canales: {signals.shape[0]}")
    print(f"Before: {segmentos['before'].shape}")
    print(f"Crisis: {segmentos['crisis'].shape}")
    print(f"After: {segmentos['after'].shape}")
    print(f"Total: {segmentos['total'].shape}")
