"""
Algorítmica y Lógica Computacional — Práctica 2
Escenario 2: Bloque Total con Ventana Deslizante

La señal completa [Before | Crisis | After] se analiza moviéndose en pasos de
frecuencia de muestreo (1 segundo). Cada ventana produce un valor de descriptor
→ se obtiene una serie temporal de descriptores que permite detectar visualmente
cuándo comienza y termina la crisis, y estimar el retardo de detección.

# pip install pyedflib numpy scipy matplotlib
"""

import numpy as np
from scipy import signal
from pyedflib import highlevel
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# =============================================================================
# PARÁMETROS (mismos que arch1.py)
# =============================================================================
EDF_FILE   = './archivos/chb20_12.edf'
START_SEC  = 94          # Segundo donde inicia la crisis
END_SEC    = 123         # Segundo donde termina la crisis
WINDOW_SEC = 120         # Contexto antes y después (2 minutos)

# Tamaño de la ventana deslizante en segundos.
# Con WIN_SEC=1 nos movemos exactamente en pasos de 1 segundo (= fs muestras).
# Se puede agrandar (e.g. 2 o 5 seg) para suavizar los descriptores.
WIN_SEC = 1

# =============================================================================
# CARGA DEL ARCHIVO EDF
# =============================================================================
signals, signal_headers, header = highlevel.read_edf(EDF_FILE)
fs = 256   # Frecuencia de muestreo (Hz) — hardcodeada igual que arch1.py

# Convertir tiempos a muestras
start_sample  = START_SEC  * fs
end_sample    = END_SEC    * fs
window_samples = WINDOW_SEC * fs
total_samples  = signals.shape[1]

inicio_before = max(0, start_sample  - window_samples)
fin_after     = min(total_samples, end_sample + window_samples)

# =============================================================================
# SEGMENTACIÓN Y CENTRADO (idéntico al Escenario 1)
# =============================================================================
def extract_and_center(sig, s, e):
    """Extrae [s:e] de todos los canales y resta la media por canal (elimina DC offset)."""
    segment = sig[:, s:e]
    return segment - segment.mean(axis=1, keepdims=True)

before_centered  = extract_and_center(signals, inicio_before, start_sample)
seizure_centered = extract_and_center(signals, start_sample,  end_sample)
after_centered   = extract_and_center(signals, end_sample,    fin_after)

# ---- Bloque total: [Before | Crisis | After] concatenados ----
total_block = np.concatenate(
    (before_centered, seizure_centered, after_centered), axis=1
)
n_canales, total_len = total_block.shape

print(f"Frecuencia de muestreo : {fs} Hz")
print(f"Bloque total           : {total_block.shape}  "
      f"({total_len / fs:.1f} seg totales)")
print(f"Inicio crisis (muestra): {start_sample - inicio_before}  "
      f"({(start_sample - inicio_before) / fs:.1f} seg dentro del bloque)")
print(f"Fin   crisis (muestra) : {end_sample   - inicio_before}  "
      f"({(end_sample   - inicio_before) / fs:.1f} seg dentro del bloque)")

# =============================================================================
# POSICIONES DE REFERENCIA DENTRO DEL BLOQUE TOTAL
# (para marcar la crisis en los gráficos)
# =============================================================================
# Cuántas muestras hay de "before" antes de la crisis
offset_before = start_sample - inicio_before

seizure_start_in_block = offset_before                    # muestra donde empieza la crisis
seizure_end_in_block   = offset_before + (end_sample - start_sample)  # muestra donde termina

# En segundos (para ejes temporales)
t_seizure_start = seizure_start_in_block / fs
t_seizure_end   = seizure_end_in_block   / fs

# =============================================================================
# VENTANA DESLIZANTE — calcula descriptores moviéndose de 1 segundo en 1 segundo
# =============================================================================
win_len  = WIN_SEC * fs   # longitud de la ventana en muestras
step     = fs             # paso = 1 segundo = fs muestras  (ver consigna: "pasos de fs")

# Calculamos cuántas ventanas enteras caben en el bloque total
n_ventanas = (total_len - win_len) // step + 1

# Arrays donde se guarda el descriptor de CADA CANAL en CADA VENTANA
# shape: (n_canales, n_ventanas)
var_timeline      = np.zeros((n_canales, n_ventanas))
std_timeline      = np.zeros((n_canales, n_ventanas))
abs_mean_timeline = np.zeros((n_canales, n_ventanas))
mean_timeline     = np.zeros((n_canales, n_ventanas))

# Para autocorrelación guardamos solo el valor en lag=0 (energía de la autocorr)
# y en lag=1 segundo (memoria de la señal)
autocorr_lag0_timeline = np.zeros((n_canales, n_ventanas))
autocorr_lag1_timeline = np.zeros((n_canales, n_ventanas))  # lag = fs muestras

# Correlación de Pearson entre canal 0 y canal 1 en cada ventana
pearson_c0c1_timeline  = np.zeros(n_ventanas)

# Tiempo central de cada ventana (en segundos, para el eje x)
t_windows = np.array([(i * step + win_len // 2) / fs for i in range(n_ventanas)])

print(f"\nVentana deslizante: {WIN_SEC} seg  |  Paso: 1 seg  |  Total ventanas: {n_ventanas}")
print("Calculando descriptores por ventana...")

for i in range(n_ventanas):
    s = i * step
    e = s + win_len
    win = total_block[:, s:e]          # shape: (n_canales, win_len)

    var_timeline[:, i]      = np.var(win,   axis=1)
    std_timeline[:, i]      = np.std(win,   axis=1)
    abs_mean_timeline[:, i] = np.mean(np.abs(win), axis=1)
    mean_timeline[:, i]     = np.mean(win,  axis=1)

    # Autocorrelación del canal 0: lag 0 (energía) y lag 1 seg
    ac = signal.correlate(win[0], win[0], mode='full')
    mid = len(ac) // 2
    autocorr_lag0_timeline[0, i] = ac[mid]            # lag = 0
    # lag de 1 segundo (= fs muestras): si existe dentro del array
    lag1_idx = mid + fs
    autocorr_lag1_timeline[0, i] = ac[lag1_idx] if lag1_idx < len(ac) else 0.0

    # Pearson entre canal 0 y canal 1
    pearson_c0c1_timeline[i] = np.corrcoef(win[0], win[1])[0, 1]

print("Listo.")

# =============================================================================
# ESCENARIO 2 — PASO 1 Y 2:
# Graficar descriptores en el tiempo + umbral dinámico + retardo de detección
# =============================================================================
def escenario2_descriptores():
    """
    Grafica varianza, desviación estándar y media absoluta de TODOS los canales
    a lo largo del tiempo, superpone el umbral [min, max] calculado sobre la
    región 'before', y marca el inicio/fin de la crisis.

    Para cada descriptor muestra también el retardo de detección estimado:
    el primer momento en que algún canal supera el umbral.
    """
    descriptores = {
        'VAR (Varianza)'                : var_timeline,
        'STD (Desviación estándar)'     : std_timeline,
        'ABS_MEAN (Media valor absoluto)': abs_mean_timeline,
    }

    # Región "before" dentro de t_windows → ventanas cuyo centro cae antes de la crisis
    before_mask = t_windows < t_seizure_start

    fig, axes = plt.subplots(len(descriptores), 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        'Escenario 2 — Descriptores estadísticos a lo largo del tiempo\n'
        '(ventana deslizante de 1 seg, todos los canales)',
        fontsize=13, fontweight='bold'
    )

    for ax, (nombre, data) in zip(axes, descriptores.items()):
        # Umbral dinámico: [min, max] del descriptor en la región "before"
        umbral_min = data[:, before_mask].min()
        umbral_max = data[:, before_mask].max()

        # Graficamos cada canal como una línea semitransparente
        for ch in range(n_canales):
            ax.plot(t_windows, data[ch], color='steelblue', alpha=0.25, linewidth=0.7)

        # Banda del umbral "normal"
        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25,
                   label=f'Rango normal [{umbral_min:.2f}, {umbral_max:.2f}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axhline(umbral_min, color='orange', linestyle='--', linewidth=1.2)

        # Marcas de inicio y fin de crisis
        ax.axvline(t_seizure_start, color='red',   linestyle='-',  linewidth=2, label='Inicio crisis')
        ax.axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')

        # Retardo de detección: primer t donde ALGÚN canal supera el umbral_max
        # (solo buscamos después de que empieza la crisis)
        post_onset_mask = t_windows >= t_seizure_start
        over_threshold  = np.any(data[:, post_onset_mask] > umbral_max, axis=0)
        t_post_onset    = t_windows[post_onset_mask]

        if over_threshold.any():
            t_deteccion = t_post_onset[np.argmax(over_threshold)]
            retardo_seg = t_deteccion - t_seizure_start
            ax.axvline(t_deteccion, color='purple', linestyle=':', linewidth=1.8,
                       label=f'Detección (retardo ≈ {retardo_seg:.1f} s)')

        ax.set_ylabel(nombre, fontsize=9)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(axis='both', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Tiempo (segundos)', fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ESCENARIO 2 — AUTOCORRELACIÓN Y PEARSON EN EL TIEMPO
# =============================================================================
def escenario2_correlaciones():
    """
    Grafica la autocorrelación (lag 0 y lag 1 seg) del canal 0
    y la correlación de Pearson entre canal 0 y canal 1 en el tiempo.
    Útil para ver la sincronización durante la crisis.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        'Escenario 2 — Autocorrelación y Correlación de Pearson en el tiempo\n'
        '(canal 0 y par canal 0–canal 1)',
        fontsize=13, fontweight='bold'
    )

    # Umbral before para autocorr lag0
    before_mask = t_windows < t_seizure_start
    umbral_ac0_max = autocorr_lag0_timeline[0, before_mask].max()
    umbral_ac0_min = autocorr_lag0_timeline[0, before_mask].min()

    # --- Autocorr lag 0 (energía) ---
    axes[0].plot(t_windows, autocorr_lag0_timeline[0], color='steelblue', linewidth=1)
    axes[0].axhspan(umbral_ac0_min, umbral_ac0_max, color='gold', alpha=0.25,
                    label=f'Rango normal [{umbral_ac0_min:.0f}, {umbral_ac0_max:.0f}]')
    axes[0].axvline(t_seizure_start, color='red',    linestyle='-', linewidth=2, label='Inicio crisis')
    axes[0].axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')
    axes[0].set_ylabel('Autocorr lag=0 (Canal 0)', fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(linestyle=':', alpha=0.4)

    # --- Autocorr lag 1 seg ---
    umbral_ac1_max = autocorr_lag1_timeline[0, before_mask].max()
    umbral_ac1_min = autocorr_lag1_timeline[0, before_mask].min()
    axes[1].plot(t_windows, autocorr_lag1_timeline[0], color='seagreen', linewidth=1)
    axes[1].axhspan(umbral_ac1_min, umbral_ac1_max, color='gold', alpha=0.25,
                    label=f'Rango normal [{umbral_ac1_min:.0f}, {umbral_ac1_max:.0f}]')
    axes[1].axvline(t_seizure_start, color='red',    linestyle='-', linewidth=2, label='Inicio crisis')
    axes[1].axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')
    axes[1].set_ylabel('Autocorr lag=1s (Canal 0)', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle=':', alpha=0.4)

    # --- Pearson canal 0 vs canal 1 ---
    umbral_p_max = pearson_c0c1_timeline[before_mask].max()
    umbral_p_min = pearson_c0c1_timeline[before_mask].min()
    axes[2].plot(t_windows, pearson_c0c1_timeline, color='tomato', linewidth=1)
    axes[2].axhspan(umbral_p_min, umbral_p_max, color='gold', alpha=0.25,
                    label=f'Rango normal [{umbral_p_min:.2f}, {umbral_p_max:.2f}]')
    axes[2].axvline(t_seizure_start, color='red',    linestyle='-', linewidth=2, label='Inicio crisis')
    axes[2].axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')
    axes[2].set_ylabel('Pearson C0–C1', fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Tiempo (segundos)', fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ESCENARIO 2 — PASO 4:
# Histograma + PDF + Diagrama de Cajas + Scatter Plot
# (mismo análisis que Escenario 1 pero sobre el bloque total completo)
# =============================================================================
def escenario2_histograma_cajas():
    """
    Para el bloque total:
      - Diagrama de cajas comparando antes / crisis / después (igual que Escenario 1
        pero acá lo mostramos sobre el total_block para apreciar la diferencia en
        una sola visualización unificada).
      - Histograma + ajuste de PDF Normal y t-location-scale sobre el bloque total
        completo del canal 0.
      - Scatter plot de parámetros (μ, σ) para cada ventana temporal, coloreado
        según si la ventana cae en zona 'before', 'seizure' o 'after'.
    """
    canal = 0

    # Señal completa del canal 0 (usado para histograma)
    data_total = total_block[canal, :]

    # Clasificamos cada ventana temporal en before / seizure / after
    labels_ventana = []
    for t_w in t_windows:
        if t_w < t_seizure_start:
            labels_ventana.append('before')
        elif t_w <= t_seizure_end:
            labels_ventana.append('seizure')
        else:
            labels_ventana.append('after')
    labels_ventana = np.array(labels_ventana)

    mask_b = labels_ventana == 'before'
    mask_s = labels_ventana == 'seizure'
    mask_a = labels_ventana == 'after'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Escenario 2 — Histograma, PDF y Scatter (bloque total)',
                 fontsize=13, fontweight='bold')

    # --- A. DIAGRAMA DE CAJAS ---
    # Usamos las muestras crudas de cada segmento (como en Escenario 1)
    n_min = min(
        before_centered.shape[1],
        seizure_centered.shape[1],
        after_centered.shape[1]
    )
    axes[0].boxplot(
        [before_centered[canal, :n_min],
         seizure_centered[canal, :n_min],
         after_centered[canal,  :n_min]],
        labels=['Antes', 'Crisis', 'Después'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='steelblue'),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[0].set_title(f'Diagrama de Cajas — Canal {canal}', fontweight='bold')
    axes[0].set_ylabel('Amplitud (μV)')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    # --- B. HISTOGRAMA + PDF sobre el bloque total del canal 0 ---
    counts, bins, _ = axes[1].hist(
        data_total, bins=60, density=True,
        alpha=0.45, color='slategray', label='Datos totales (Canal 0)'
    )
    # Ajuste Normal
    mu_n, std_n = norm.fit(data_total)
    p_norm = norm.pdf(bins, mu_n, std_n)
    axes[1].plot(bins, p_norm, 'k--', linewidth=2,
                 label=f'Normal (μ={mu_n:.2f}, σ={std_n:.2f})')

    # Ajuste t-location-scale (recomendado en la teoría para señales EEG)
    df_t, loc_t, scale_t = t.fit(data_total)
    p_t = t.pdf(bins, df_t, loc_t, scale_t)
    axes[1].plot(bins, p_t, 'b-', linewidth=2,
                 label=f't-loc-scale (df={df_t:.1f})')

    axes[1].set_title('Histograma + PDF — Bloque Total', fontweight='bold')
    axes[1].set_xlabel('Amplitud (μV)')
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle=':', alpha=0.4)

    # --- C. SCATTER PLOT de (μ_ventana, σ_ventana) coloreado por etapa ---
    # Cada punto es UNA VENTANA TEMPORAL del canal 0
    mu_v  = mean_timeline[canal, :]
    std_v = std_timeline[canal, :]

    axes[2].scatter(mu_v[mask_b], std_v[mask_b], color='steelblue',
                    label='Before (normal)', alpha=0.6, s=18)
    axes[2].scatter(mu_v[mask_s], std_v[mask_s], color='tomato',
                    label='Crisis', alpha=0.8, s=30)
    axes[2].scatter(mu_v[mask_a], std_v[mask_a], color='seagreen',
                    label='After', alpha=0.6, s=18)

    axes[2].set_title('Scatter μ vs σ por ventana — Canal 0', fontweight='bold')
    axes[2].set_xlabel('Media (μ)')
    axes[2].set_ylabel('Desviación estándar (σ)')
    axes[2].legend(fontsize=9)
    axes[2].grid(linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()



# =============================================================================
# EJECUCIÓN
# =============================================================================
escenario2_descriptores()      # Paso 1 y 2: descriptores en tiempo + umbral + retardo
escenario2_correlaciones()     # Paso 1: autocorrelación y Pearson en tiempo
escenario2_histograma_cajas()  # Paso 4: histograma, PDF, cajas, scatter

