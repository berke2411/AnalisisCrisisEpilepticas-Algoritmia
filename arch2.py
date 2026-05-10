"""
Algorítmica y Lógica Computacional — Práctica 2
Escenario 2: Bloque Total con Ventana Deslizante

La señal completa [Before | Crisis | After] se analiza moviéndose en pasos de
1 segundo (= fs muestras). El umbral aplicado proviene del análisis por bloques
del Escenario 1 (arch1.py), usando el descriptor con mayor poder discriminante.

# pip install pyedflib numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# =============================================================================
# IMPORTACIONES DESDE ESCENARIO 1
# Se reutilizan los datos ya cargados/centrados y el umbral hallado por bloques.
# =============================================================================
from arch1 import (
    BEST_DESCRIPTOR, UMBRAL_MIN_E1, UMBRAL_MAX_E1,
    before_centered, seizure_centered, after_centered,
    total_block, n_canales, fs,
    start_sample, end_sample, inicio_before,
    stats,
)

# =============================================================================
# POSICIONES DE REFERENCIA DENTRO DEL BLOQUE TOTAL
# =============================================================================
offset_before          = start_sample - inicio_before
seizure_start_in_block = offset_before
seizure_end_in_block   = offset_before + (end_sample - start_sample)
t_seizure_start        = seizure_start_in_block / fs
t_seizure_end          = seizure_end_in_block   / fs
total_len              = total_block.shape[1]

print(f"\n{'='*60}")
print(f"ESCENARIO 2 — Bloque total: {total_block.shape}  ({total_len / fs:.1f} seg)")
print(f"Crisis: {t_seizure_start:.1f}s – {t_seizure_end:.1f}s")
print(f"Descriptor elegido en E1 : {BEST_DESCRIPTOR.upper()}")
print(f"Umbral E1                : [{UMBRAL_MIN_E1:.4f}, {UMBRAL_MAX_E1:.4f}]")
print(f"{'='*60}")

# =============================================================================
# UMBRALES E1 PARA TODOS LOS DESCRIPTORES
# Derivados del bloque completo Before en el Escenario 1 (no de ventanas).
# =============================================================================
E1_UMBRALES = {
    'var':      (float(stats['before']['var'].min()),      float(stats['before']['var'].max())),
    'std':      (float(stats['before']['std'].min()),      float(stats['before']['std'].max())),
    'abs_mean': (float(stats['before']['abs_mean'].min()), float(stats['before']['abs_mean'].max())),
}

# =============================================================================
# VENTANA DESLIZANTE — pasos de 1 segundo (= fs muestras)
# =============================================================================
WIN_SEC      = 1   # duración de la ventana deslizante en segundos
PERSIST_SEC  = 3   # ventanas consecutivas requeridas para confirmar detección
MIN_CHANNELS = 3   # canales que deben superar el umbral simultáneamente
win_len    = WIN_SEC * fs
step       = fs
n_ventanas = (total_len - win_len) // step + 1

var_timeline           = np.zeros((n_canales, n_ventanas))
std_timeline           = np.zeros((n_canales, n_ventanas))
abs_mean_timeline      = np.zeros((n_canales, n_ventanas))
mean_timeline          = np.zeros((n_canales, n_ventanas))
autocorr_lag0_timeline = np.zeros((n_canales, n_ventanas))  # energía lag=0, todos los canales
mean_pearson_timeline  = np.zeros(n_ventanas)               # sincronía media entre todos los pares
frob_cov_timeline      = np.zeros(n_ventanas)               # norma Frobenius off-diagonal de covarianza

# Máscara reutilizable: True donde i≠j (pares distintos de canales)
mask_offdiag = ~np.eye(n_canales, dtype=bool)

t_windows = np.array([(i * step + win_len // 2) / fs for i in range(n_ventanas)])

print(f"\nVentana deslizante: {WIN_SEC} seg  |  Paso: 1 seg  |  Total ventanas: {n_ventanas}")
print("Calculando descriptores por ventana...")

for i in range(n_ventanas):
    s   = i * step
    e   = s + win_len
    win = total_block[:, s:e]

    var_timeline[:, i]      = np.var(win,          axis=1)
    std_timeline[:, i]      = np.std(win,          axis=1)
    abs_mean_timeline[:, i] = np.mean(np.abs(win), axis=1)
    mean_timeline[:, i]     = np.mean(win,         axis=1)

    # Autocorrelación lag=0: sum(x²) por canal = energía, todos los canales
    autocorr_lag0_timeline[:, i] = np.sum(win ** 2, axis=1)

    # Pearson N×N → escalar: media |off-diagonal| = sincronía media entre todos los pares
    R = np.corrcoef(win)
    mean_pearson_timeline[i] = np.nanmean(np.abs(R[mask_offdiag]))

    # Covarianza N×N → escalar: norma Frobenius de la parte off-diagonal
    C = np.cov(win)
    C_offdiag = C.copy()
    np.fill_diagonal(C_offdiag, 0)
    frob_cov_timeline[i] = np.linalg.norm(C_offdiag, 'fro')

print("Listo.")

DESCRIPTOR_MAP = {
    'var':      var_timeline,
    'std':      std_timeline,
    'abs_mean': abs_mean_timeline,
}
DESCRIPTOR_NAMES = {
    'var':      'VAR (Varianza)',
    'std':      'STD (Desviación estándar)',
    'abs_mean': 'ABS_MEAN (Media valor absoluto)',
}

def _detect_persistent(data, umbral_max, persist_sec=PERSIST_SEC, min_channels=MIN_CHANNELS):
    """
    Busca la primera racha de 'persist_sec' ventanas consecutivas en las que
    al menos 'min_channels' canales superen 'umbral_max'.
    Retorna el índice de la primera ventana de la racha, o None si no hay detección.
    Busca en toda la señal (sin conocer cuándo empieza la crisis).
    """
    active = np.sum(data > umbral_max, axis=0) >= min_channels
    count  = 0
    for i, a in enumerate(active):
        if a:
            count += 1
            if count >= persist_sec:
                return i - persist_sec + 1
        else:
            count = 0
    return None


# =============================================================================
# ESCENARIO 2 — PASO 1 Y 2:
# Descriptores en el tiempo + umbral E1 + retardo de detección
# =============================================================================
def escenario2_descriptores():
    """
    Pre-computa el retardo de detección para los tres descriptores usando el umbral
    de E1 y el criterio de persistencia. Determina el mejor descriptor para detección
    (mínimo retardo) y lo compara con el mejor de E1 (máximo ratio). Grafica los tres.
    """
    # --- Paso 1: pre-computar retardos para los tres descriptores ---
    retardos  = {}
    idx_detec = {}
    for d, data in DESCRIPTOR_MAP.items():
        _, umbral_max = E1_UMBRALES[d]
        idx = _detect_persistent(data, umbral_max)
        if idx is not None:
            t_dec       = t_windows[min(idx + PERSIST_SEC - 1, n_ventanas - 1)]
            retardos[d] = t_dec - t_seizure_start
        else:
            retardos[d] = np.inf
        idx_detec[d] = idx

    # Descriptor con menor retardo → mejor para detección en tiempo real
    best_detector = min(retardos, key=retardos.get)

    # --- Tabla comparativa en consola ---
    print(f"\n{'='*65}")
    print(f"  COMPARATIVA E1 (discriminabilidad) vs E2 (retardo de detección)")
    print(f"  Criterio persistencia: {PERSIST_SEC}s consecutivos, ≥{MIN_CHANNELS} canales")
    print(f"{'─'*65}")
    print(f"  {'Descriptor':<12} {'Ratio E1':>10}   {'Retardo E2':>12}   Mejor para")
    print(f"{'─'*65}")
    for d in DESCRIPTOR_MAP:
        ratio      = (stats['seizure'][d] / (stats['before'][d] + 1e-10)).mean()
        ret_str    = f"{retardos[d]:.1f} s" if retardos[d] != np.inf else "no detecta"
        etiquetas  = []
        if d == BEST_DESCRIPTOR: etiquetas.append('← E1 discriminabilidad')
        if d == best_detector:   etiquetas.append('← E2 detección')
        print(f"  {d.upper():<12} {ratio:>8.2f}x   {ret_str:>10}   {'  '.join(etiquetas)}")
    print(f"{'='*65}\n")

    # --- Paso 2: graficar los tres descriptores ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        'Escenario 2 — Descriptores estadísticos a lo largo del tiempo\n'
        f'E1 (discriminabilidad): {BEST_DESCRIPTOR.upper()}  |  '
        f'E2 (detección): {best_detector.upper()}',
        fontsize=13, fontweight='bold'
    )

    for ax, (d, data) in zip(axes, DESCRIPTOR_MAP.items()):
        _, umbral_max = E1_UMBRALES[d]
        umbral_min, _ = E1_UMBRALES[d]
        es_mejor_e1   = (d == BEST_DESCRIPTOR)
        es_mejor_e2   = (d == best_detector)

        nombre = DESCRIPTOR_NAMES[d]
        if es_mejor_e1 and es_mejor_e2:
            nombre += '  ← E1 y E2'
        elif es_mejor_e1:
            nombre += '  ← E1 (discriminabilidad)'
        elif es_mejor_e2:
            nombre += '  ← E2 (detección)'

        # Color de líneas: verde para el mejor detector, rojo para E1, azul para el resto
        if es_mejor_e2:
            color_ch = 'seagreen'
        elif es_mejor_e1:
            color_ch = 'tomato'
        else:
            color_ch = 'steelblue'

        for ch in range(n_canales):
            ax.plot(t_windows, data[ch], color=color_ch, alpha=0.25, linewidth=0.7)

        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25,
                   label=f'Umbral E1 [{umbral_min:.2f}, {umbral_max:.2f}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axhline(umbral_min, color='orange', linestyle='--', linewidth=1.2)
        ax.axvline(t_seizure_start, color='red',     linestyle='-', linewidth=2, label='Inicio crisis')
        ax.axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')

        idx = idx_detec[d]
        if idx is not None:
            t_inicio_racha = t_windows[idx]
            t_decision     = t_windows[min(idx + PERSIST_SEC - 1, n_ventanas - 1)]
            retardo        = retardos[d]
            estado         = 'CORRECTO' if t_decision >= t_seizure_start else 'FALSO POSITIVO'
            ax.axvline(t_inicio_racha, color='purple', linestyle='--', linewidth=1.2,
                       label=f'Inicio racha ({PERSIST_SEC}s, ≥{MIN_CHANNELS} ch)')
            ax.axvline(t_decision, color='indigo', linestyle=':', linewidth=1.8,
                       label=f'Decisión {estado} (retardo={retardo:.1f}s)')

        fw = 'bold' if (es_mejor_e1 or es_mejor_e2) else 'normal'
        ax.set_ylabel(nombre, fontsize=9, fontweight=fw)
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
    Grafica la energía de autocorrelación (lag=0) promediada sobre TODOS los canales
    y la correlación de Pearson media inter-canal, ambas con criterio de persistencia
    para detectar la crisis. Mismo formato que escenario2_descriptores().
    """
    before_mask        = t_windows < t_seizure_start
    autocorr_lag0_mean = autocorr_lag0_timeline.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f'Escenario 2 — Autocorrelación y Pearson inter-canal en el tiempo\n'
        f'(todos los {n_canales} canales  |  criterio: {PERSIST_SEC}s consecutivos, ≥{MIN_CHANNELS} canales)',
        fontsize=13, fontweight='bold'
    )

    metricas = [
        (autocorr_lag0_mean,    'Energía Autocorr lag=0 (media todos los canales)', 'steelblue', '.0f'),
        (mean_pearson_timeline, 'Pearson media inter-canal (todos los pares)',       'tomato',    '.3f'),
    ]

    for ax, (timeline, ylabel, color, fmt) in zip(axes, metricas):
        umbral_min = timeline[before_mask].min()
        umbral_max = timeline[before_mask].max()

        ax.plot(t_windows, timeline, color=color, linewidth=1.2)
        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25,
                   label=f'Rango normal Before [{umbral_min:{fmt}}, {umbral_max:{fmt}}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axvline(t_seizure_start, color='red',     linestyle='-', linewidth=2, label='Inicio crisis')
        ax.axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')

        idx = _detect_persistent(timeline.reshape(1, -1), umbral_max, min_channels=1)
        if idx is not None:
            t_inicio = t_windows[idx]
            t_dec    = t_windows[min(idx + PERSIST_SEC - 1, n_ventanas - 1)]
            retardo  = t_dec - t_seizure_start
            estado   = 'CORRECTO' if t_dec >= t_seizure_start else 'FALSO POSITIVO'
            ax.axvline(t_inicio, color='purple', linestyle='--', linewidth=1.2,
                       label=f'Inicio racha ({PERSIST_SEC}s)')
            ax.axvline(t_dec, color='indigo', linestyle=':', linewidth=1.8,
                       label=f'Decisión {estado} (retardo={retardo:.1f}s)')

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(axis='both', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Tiempo (segundos)', fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ESCENARIO 2 — SINCRONÍA INTER-CANAL (Pearson media + Frobenius covarianza)
# =============================================================================
def escenario2_sincronia():
    """
    Grafica las métricas de sincronía entre TODOS los canales a lo largo del tiempo:
      - mean_pearson_timeline : media del |coef. de Pearson| de todos los pares i≠j.
        En reposo los canales son relativamente independientes (valor bajo).
        Durante la crisis se sincronizan → valor sube fuertemente.
      - frob_cov_timeline     : norma Frobenius de la parte off-diagonal de la matriz
        de covarianza. Mide la covarianza total entre pares distintos de canales.
    Ambas métricas se evalúan con el criterio de persistencia para detectar la crisis.
    El umbral se deriva de la región Before del propio bloque deslizante, ya que estas
    métricas producen un escalar único por bloque en E1 (sin rango [min,max]).
    """
    before_mask = t_windows < t_seizure_start

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        'Escenario 2 — Sincronía Inter-Canal a lo largo del tiempo\n'
        f'(todos los {n_canales} canales  |  {n_canales*(n_canales-1)//2} pares  |'
        f'  criterio: {PERSIST_SEC}s consecutivos)',
        fontsize=13, fontweight='bold'
    )

    metricas = [
        (mean_pearson_timeline, 'Pearson media inter-canal  |r̄|',          'tomato',    '.3f'),
        (frob_cov_timeline,     'Frobenius off-diag covarianza',            'steelblue', '.0f'),
    ]

    for ax, (timeline, ylabel, color, fmt) in zip(axes, metricas):
        umbral_min = timeline[before_mask].min()
        umbral_max = timeline[before_mask].max()

        ax.plot(t_windows, timeline, color=color, linewidth=1.2)
        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25,
                   label=f'Rango normal Before [{umbral_min:{fmt}}, {umbral_max:{fmt}}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axvline(t_seizure_start, color='red',     linestyle='-', linewidth=2, label='Inicio crisis')
        ax.axvline(t_seizure_end,   color='darkred', linestyle='-', linewidth=2, label='Fin crisis')

        # Detección con persistencia sobre métrica escalar (reshape para reutilizar _detect_persistent)
        idx = _detect_persistent(timeline.reshape(1, -1), umbral_max, min_channels=1)
        if idx is not None:
            t_inicio = t_windows[idx]
            t_dec    = t_windows[min(idx + PERSIST_SEC - 1, n_ventanas - 1)]
            retardo  = t_dec - t_seizure_start
            estado   = 'CORRECTO' if t_dec >= t_seizure_start else 'FALSO POSITIVO'
            ax.axvline(t_inicio, color='purple', linestyle='--', linewidth=1.2,
                       label=f'Inicio racha ({PERSIST_SEC}s)')
            ax.axvline(t_dec, color='indigo', linestyle=':', linewidth=1.8,
                       label=f'Decisión {estado} (retardo={retardo:.1f}s)')

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(axis='both', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Tiempo (segundos)', fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ESCENARIO 2 — PASO 4: Histograma + PDF + Diagrama de Cajas + Scatter
# =============================================================================
def escenario2_histograma_cajas():
    """
    Histograma + PDF sobre el bloque total del canal 0.
    Diagrama de cajas comparando antes / crisis / después.
    Scatter de (μ, σ) por ventana temporal coloreado por etapa.
    """
    canal = 0

    labels_ventana = np.array([
        'before' if t < t_seizure_start else ('seizure' if t <= t_seizure_end else 'after')
        for t in t_windows
    ])
    mask_b = labels_ventana == 'before'
    mask_s = labels_ventana == 'seizure'
    mask_a = labels_ventana == 'after'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Escenario 2 — Histograma, PDF y Scatter (bloque total)',
                 fontsize=13, fontweight='bold')

    # A. Diagrama de cajas
    n_min = min(before_centered.shape[1], seizure_centered.shape[1], after_centered.shape[1])
    axes[0].boxplot(
        [before_centered[canal, :n_min],
         seizure_centered[canal, :n_min],
         after_centered[canal,   :n_min]],
        labels=['Antes', 'Crisis', 'Después'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='steelblue'),
        medianprops=dict(color='red', linewidth=2)
    )
    axes[0].set_title(f'Diagrama de Cajas — Canal {canal}', fontweight='bold')
    axes[0].set_ylabel('Amplitud (μV)')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    # B. Histograma + PDF sobre el bloque total
    data_total = total_block[canal, :]
    counts, bins, _ = axes[1].hist(
        data_total, bins=60, density=True,
        alpha=0.45, color='slategray', label='Datos totales (Canal 0)'
    )
    mu_n, std_n = norm.fit(data_total)
    axes[1].plot(bins, norm.pdf(bins, mu_n, std_n), 'k--', linewidth=2,
                 label=f'Normal (μ={mu_n:.2f}, σ={std_n:.2f})')
    df_t, loc_t, scale_t = t.fit(data_total)
    axes[1].plot(bins, t.pdf(bins, df_t, loc_t, scale_t), 'b-', linewidth=2,
                 label=f't-loc-scale (df={df_t:.1f})')
    axes[1].set_title('Histograma + PDF — Bloque Total', fontweight='bold')
    axes[1].set_xlabel('Amplitud (μV)')
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle=':', alpha=0.4)

    # C. Scatter (μ, σ) por ventana coloreado por etapa
    mu_v  = mean_timeline[canal, :]
    std_v = std_timeline[canal, :]
    axes[2].scatter(mu_v[mask_b], std_v[mask_b], color='steelblue', label='Before', alpha=0.6, s=18)
    axes[2].scatter(mu_v[mask_s], std_v[mask_s], color='tomato',    label='Crisis',  alpha=0.8, s=30)
    axes[2].scatter(mu_v[mask_a], std_v[mask_a], color='seagreen',  label='After',   alpha=0.6, s=18)
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
escenario2_descriptores()      # Paso 1 y 2: VAR/STD/ABS_MEAN con umbral E1 + retardo
escenario2_sincronia()         # Paso 1 y 2: Pearson media y Frobenius covarianza inter-canal
escenario2_correlaciones()     # Paso 1: autocorr (todos canales) y Pearson media en tiempo
escenario2_histograma_cajas()  # Paso 4: histograma, PDF, cajas, scatter
