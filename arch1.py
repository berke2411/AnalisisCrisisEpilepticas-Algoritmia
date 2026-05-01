import numpy as np
from scipy import signal
from pyedflib import highlevel


# --- Parámetros ---
EDF_FILE = './archivos/chb20_12.edf'
START_SEC = 94
END_SEC = 123
WINDOW_SEC = 120

# --- Carga ---
signals, signal_headers, header = highlevel.read_edf(EDF_FILE)
fs = 256

# --- Muestras ---
start_sample = START_SEC * fs
end_sample = END_SEC * fs
window_samples = WINDOW_SEC * fs
total_samples = signals.shape[1]

inicio_before = max(0, start_sample - window_samples)
fin_after = min(total_samples, end_sample + window_samples)

# --- Segmentación y centrado ---
def extract_and_center(sig, s, e):
    segment = sig[:, s:e]
    return segment - segment.mean(axis=1, keepdims=True)

before_centered = extract_and_center(signals, inicio_before, start_sample)
seizure_centered = extract_and_center(signals, start_sample, end_sample)
after_centered   = extract_and_center(signals, end_sample, fin_after)

total_block = np.concatenate(
    (before_centered, seizure_centered, after_centered), axis=1
)

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Dimensiones del bloque 'Before': {before_centered.shape}")
print(f"Dimensiones del bloque 'Crisis': {seizure_centered.shape}")
print(f"Dimensiones del bloque 'After': {after_centered.shape}")
print(f"Dimensiones del 'Bloque Total': {total_block.shape}")

# --- Descriptores ---
def compute_stats(seg):
    var = np.var(seg, axis=1)
    return {
        'var':      var,
        'std':      np.sqrt(var),
        'abs_mean': np.mean(np.abs(seg), axis=1),
        'cov':      np.cov(seg),
        'pearson':  np.corrcoef(seg),
    }

segments = {
    'before':  before_centered,
    'seizure': seizure_centered,
    'after':   after_centered,
}
stats = {name: compute_stats(seg) for name, seg in segments.items()}

# --- Autocorrelación ---
C0 = 0
autocorr = {
    name: signal.correlate(seg[C0], seg[C0], mode='full')
    for name, seg in segments.items()
}

# --- Reporte ---
C0, C1 = 0, 1
print(f"fs: {fs} Hz | Bloque total: {total_block.shape}")
for metric in ('var', 'std', 'abs_mean'):
    vals = " | ".join(
        f"{name}: {stats[name][metric][C0]:.2f}"
        for name in ('before', 'seizure', 'after')
    )
    print(f"{metric.upper():12} -> {vals}")

print(f"\nCovarianza C{C0}-C{C1}  -> "
      f"Antes: {stats['before']['cov'][C0, C1]:.2f} | "
      f"Crisis: {stats['seizure']['cov'][C0, C1]:.2f}")
print(f"Pearson C{C0}-C{C1}     -> "
      f"Antes: {stats['before']['pearson'][C0, C1]:.2f} | "
      f"Crisis: {stats['seizure']['pearson'][C0, C1]:.2f}")