import numpy as np
from scipy import signal
from pyedflib import highlevel
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# --- Parámetros ---
EDF_FILE   = './archivos/chb20_12.edf'
START_SEC  = 94    # Segundo donde inicia la crisis
END_SEC    = 123   # Segundo donde termina la crisis
WINDOW_SEC = 120   # Contexto antes y después (2 minutos)

# --- Carga del archivo EDF ---
signals, signal_headers, header = highlevel.read_edf(EDF_FILE)
fs = 256  # Frecuencia de muestreo (Hz)

# --- Conversión de tiempos a muestras ---
start_sample   = START_SEC  * fs
end_sample     = END_SEC    * fs
window_samples = WINDOW_SEC * fs
total_samples  = signals.shape[1]

inicio_before = max(0, start_sample - window_samples)
fin_after     = min(total_samples, end_sample + window_samples)

# --- Segmentación y centrado ---
def extract_and_center(sig, s, e):
    """Extrae el segmento [s:e] para todos los canales y resta la media por canal (elimina DC offset)."""
    segment = sig[:, s:e]
    return segment - segment.mean(axis=1, keepdims=True)

before_centered  = extract_and_center(signals, inicio_before, start_sample)
seizure_centered = extract_and_center(signals, start_sample, end_sample)
after_centered   = extract_and_center(signals, end_sample, fin_after)

total_block = np.concatenate(
    (before_centered, seizure_centered, after_centered), axis=1
)

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Canales: {before_centered.shape[0]}")
print(f"Dimensiones del bloque 'Before':  {before_centered.shape}")
print(f"Dimensiones del bloque 'Crisis':  {seizure_centered.shape}")
print(f"Dimensiones del bloque 'After':   {after_centered.shape}")
print(f"Dimensiones del 'Bloque Total':   {total_block.shape}")

# --- Descriptores estadísticos por bloque (todos los canales) ---
def compute_stats(seg):
    """Calcula descriptores estadísticos para todos los canales de un segmento."""
    var         = np.var(seg, axis=1)
    pearson_mat = np.corrcoef(seg)
    cov_mat     = np.cov(seg)
    cov_offdiag = cov_mat.copy()
    np.fill_diagonal(cov_offdiag, 0)
    mask_off    = ~np.eye(seg.shape[0], dtype=bool)
    return {
        'var':              var,
        'mean':             np.mean(seg, axis=1),
        'std':              np.sqrt(var),
        'abs_mean':         np.mean(np.abs(seg), axis=1),
        'cov':              cov_mat,
        'pearson':          pearson_mat,
        # escalares de sincronía inter-canal (un valor por bloque, no por canal)
        'pearson_sync':     float(np.nanmean(np.abs(pearson_mat[mask_off]))),
        'frob_cov':         float(np.linalg.norm(cov_offdiag, 'fro')),
    }

segments = {
    'before':  before_centered,
    'seizure': seizure_centered,
    'after':   after_centered,
}

stats = {name: compute_stats(seg) for name, seg in segments.items()}
n_canales = before_centered.shape[0]

# --- Reporte resumido ---
print(f"\n{'Descriptor':<16} {'Before':>12} {'Seizure':>12} {'After':>12} {'Ratio S/B':>12}")
print(f"{'─'*64}")
# Métricas por canal (promedio sobre canales)
for metric in ('var', 'std', 'abs_mean'):
    v_b = stats['before'][metric].mean()
    v_s = stats['seizure'][metric].mean()
    v_a = stats['after'][metric].mean()
    print(f"{metric.upper():<16} {v_b:>12.2f} {v_s:>12.2f} {v_a:>12.2f} {v_s / (v_b + 1e-10):>12.2f}x")
print(f"{'─'*64}")
# Escalares de sincronía inter-canal (un valor por bloque)
for metric, label in (('pearson_sync', 'PEARSON_SYNC'), ('frob_cov', 'FROB_COV')):
    v_b = stats['before'][metric]
    v_s = stats['seizure'][metric]
    v_a = stats['after'][metric]
    print(f"{label:<16} {v_b:>12.4f} {v_s:>12.4f} {v_a:>12.4f} {v_s / (v_b + 1e-10):>12.2f}x")


def _best_descriptor():
    """Calcula el mejor descriptor y umbral [min, max] sin generar gráficos.
    Exportado para que arch2.py lo importe sin disparar plots."""
    descriptores = ['var', 'std', 'abs_mean']
    mean_ratios = {}
    for d in descriptores:
        ratio = stats['seizure'][d] / (stats['before'][d] + 1e-10)
        mean_ratios[d] = ratio.mean()
    best = max(mean_ratios, key=mean_ratios.get)
    return best, float(stats['before'][best].min()), float(stats['before'][best].max())


# ==============================================================================
# ESCENARIO 1 — Análisis de Discriminabilidad
# Determina cuál descriptor separa mejor Before de Crisis en TODOS los canales.
# ==============================================================================
def analisis_discriminabilidad():
    """
    Para cada descriptor escalar (var, std, abs_mean), calcula el ratio
    seizure/before por canal. El descriptor con mayor ratio medio es el más
    útil para construir un umbral de detección.
    Retorna (mejor_descriptor, umbral_min, umbral_max).
    """
    descriptores = ['var', 'std', 'abs_mean']
    x_pos = np.arange(n_canales)

    scores      = {}
    mean_ratios = {}
    for d in descriptores:
        ratio       = stats['seizure'][d] / (stats['before'][d] + 1e-10)
        scores[d]      = ratio
        mean_ratios[d] = ratio.mean()

    best = max(mean_ratios, key=mean_ratios.get)

    print("\n--- Discriminabilidad (ratio Seizure/Before, promediado sobre todos los canales) ---")
    for d in descriptores:
        flag = " ← MEJOR" if d == best else ""
        print(f"  {d.upper():<12} ratio medio: {mean_ratios[d]:.2f}x  "
              f"| canal max: {scores[d].argmax()} ({scores[d].max():.2f}x){flag}")

    # Gráfico 1: valores absolutos de cada descriptor, Before vs Crisis vs After (todos los canales)
    fig, axes = plt.subplots(len(descriptores), 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Escenario 1 — Descriptores por Canal: Before vs Crisis vs After\n'
                 '(todos los canales, para determinar el mejor descriptor)',
                 fontsize=13, fontweight='bold')

    w = 0.28
    for ax, d in zip(axes, descriptores):
        ax.bar(x_pos - w, stats['before'][d],  width=w, color='steelblue', alpha=0.8, label='Before')
        ax.bar(x_pos,     stats['seizure'][d], width=w, color='tomato',    alpha=0.8, label='Crisis')
        ax.bar(x_pos + w, stats['after'][d],   width=w, color='seagreen',  alpha=0.8, label='After')

        ylabel = f'{d.upper()} ← MEJOR' if d == best else d.upper()
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold' if d == best else 'normal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Canal', fontsize=11)
    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(x_pos, fontsize=7)
    plt.tight_layout()
    plt.show()

    # Gráfico 2: ratio de discriminabilidad puro (seizure / before) por canal
    fig2, axes2 = plt.subplots(len(descriptores), 1, figsize=(14, 8), sharex=True)
    fig2.suptitle('Ratio de Discriminabilidad (Seizure/Before) por Canal\n'
                  '(barras rojas = canal con ratio > 1.5)',
                  fontsize=13, fontweight='bold')

    for ax, d in zip(axes2, descriptores):
        ratio  = scores[d]
        colors = ['tomato' if r > 1.5 else 'steelblue' for r in ratio]
        ax.bar(x_pos, ratio, color=colors, alpha=0.8, width=0.7)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.2, label='ratio=1 (sin cambio)')
        ax.axhline(mean_ratios[d], color='orange', linestyle='-', linewidth=1.5,
                   label=f'Media={mean_ratios[d]:.2f}x')
        ylabel = f'{d.upper()} ← MEJOR' if d == best else d.upper()
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold' if d == best else 'normal')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes2[-1].set_xlabel('Canal', fontsize=11)
    axes2[-1].set_xticks(x_pos)
    axes2[-1].set_xticklabels(x_pos, fontsize=7)
    plt.tight_layout()
    plt.show()

    umbral_min = stats['before'][best].min()
    umbral_max = stats['before'][best].max()
    print(f"\nUmbral [{best.upper()}] definido por bloque Before: [{umbral_min:.4f}, {umbral_max:.4f}]")
    return best, umbral_min, umbral_max


# ==============================================================================
# ESCENARIO 1 — PASO 2: Umbral dinámico con el mejor descriptor
# ==============================================================================
def escenario1paso2(descriptor, umbral_min, umbral_max):
    """
    Grafica el descriptor elegido para los 3 bloques mostrando el umbral [min, max]
    definido a partir del bloque Before. Canales fuera del rango se destacan en rojo.
    """
    x_pos = np.arange(n_canales)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    bloques = [
        ('Antes (Reposo)',         stats['before'][descriptor],  'steelblue'),
        ('Crisis (Epilepsia)',     stats['seizure'][descriptor], 'tomato'),
        ('Después (Recuperación)', stats['after'][descriptor],   'seagreen'),
    ]

    for ax, (titulo, vals, color) in zip(axes, bloques):
        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25, zorder=0,
                   label=f'Rango Normal [{umbral_min:.2f}, {umbral_max:.2f}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axhline(umbral_min, color='orange', linestyle='--', linewidth=1.2)

        for i, v in enumerate(vals):
            fuera = v > umbral_max or v < umbral_min
            ax.bar(i, v, color='red' if fuera else color, alpha=0.8, width=0.7, zorder=2)

        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel(descriptor.upper(), fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Canal', fontsize=11)
    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(x_pos, fontsize=7)
    fig.suptitle(f'Umbral Dinámico [min, max] del descriptor "{descriptor.upper()}"\n'
                 f'por canal — Umbral definido por bloque "Antes"',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ESCENARIO 1 — Heatmap de Correlación de Pearson (sincronía entre canales)
# ==============================================================================
def heatmap_pearson():
    """
    Muestra la matriz de correlación de Pearson entre todos los canales para
    cada uno de los 3 bloques. La sincronía aumentada durante la crisis se
    manifiesta como correlaciones más altas y uniformes en la matriz de crisis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    configs = [
        ('before',  'Before (Reposo)',      'Blues'),
        ('seizure', 'Crisis (Epilepsia)',   'Reds'),
        ('after',   'After (Recuperación)', 'Greens'),
    ]

    for ax, (name, titulo, cmap) in zip(axes, configs):
        mat = stats[name]['pearson']
        im  = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_title(titulo, fontweight='bold')
        ax.set_xlabel('Canal')
        ax.set_ylabel('Canal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Matriz de Correlación de Pearson — Todos los Canales\n'
                 '(la sincronía inter-canal aumenta notablemente durante la crisis)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ESCENARIO 1 — Heatmap de Covarianza (sincronía inter-canal en magnitud)
# ==============================================================================
def heatmap_covarianza():
    """
    Muestra la matriz de covarianza entre todos los canales para los 3 bloques.
    Complementa el heatmap de Pearson: mientras Pearson mide correlación normalizada
    (sin unidades), la covarianza preserva la magnitud de la co-variación.
    Elementos off-diagonal grandes = canales co-variando fuertemente = sincronía.
    Se usa escala simétrica común a los 3 bloques para comparación directa.
    """
    configs = [
        ('before',  'Before (Reposo)',      'Blues'),
        ('seizure', 'Crisis (Epilepsia)',   'Reds'),
        ('after',   'After (Recuperación)', 'Greens'),
    ]

    vmax = max(np.abs(stats[name]['cov']).max() for name, _, _ in configs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, titulo, cmap) in zip(axes, configs):
        mat = stats[name]['cov']
        im  = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        ax.set_title(
            f"{titulo}\nFrob off-diag = {stats[name]['frob_cov']:.1f}",
            fontweight='bold'
        )
        ax.set_xlabel('Canal')
        ax.set_ylabel('Canal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Matriz de Covarianza — Todos los Canales\n'
                 '(off-diagonal grande = canales co-variando = sincronía epiléptica)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ESCENARIO 1 — Autocorrelación para todos los canales (energía lag=0)
# ==============================================================================
def autocorr_todos_canales():
    """
    Calcula la energía de autocorrelación (valor en lag=0) para todos los canales
    en los 3 bloques. La energía en lag=0 es proporcional a la potencia de la señal;
    su aumento durante la crisis confirma la actividad neuronal excesiva.
    """
    def energia_lag0(seg):
        energia = np.zeros(seg.shape[0])
        for ch in range(seg.shape[0]):
            ac = signal.correlate(seg[ch], seg[ch], mode='full')
            energia[ch] = ac[len(ac) // 2]
        return energia

    e_before  = energia_lag0(before_centered)
    e_seizure = energia_lag0(seizure_centered)
    e_after   = energia_lag0(after_centered)

    x_pos = np.arange(n_canales)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Autocorrelación — Energía en lag=0 por Canal\n'
                 '(todos los canales, confirma actividad excesiva durante la crisis)',
                 fontsize=13, fontweight='bold')

    for ax, (titulo, energia, color) in zip(axes, [
        ('Before (Reposo)',      e_before,  'steelblue'),
        ('Crisis (Epilepsia)',   e_seizure, 'tomato'),
        ('After (Recuperación)', e_after,   'seagreen'),
    ]):
        ax.bar(x_pos, energia, color=color, alpha=0.8, width=0.7)
        ax.set_title(titulo, fontweight='bold')
        ax.set_ylabel('Energía Autocorr (lag=0)', fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Canal', fontsize=11)
    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(x_pos, fontsize=7)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# ESCENARIO 1 — PASO 4: Histograma, PDF, Boxplot, Scatter
# ==============================================================================
def escenario1paso4():
    """
    Boxplot e histograma sobre el canal 0 (caso representativo) para los 3 bloques.
    Scatter plot de (μ, σ) usa TODOS los canales en los 3 estados.
    """
    canal = 0
    data_before  = before_centered[canal, :]
    data_seizure = seizure_centered[canal, :]
    data_after   = after_centered[canal, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Escenario 1 — Histograma, PDF, Boxplot y Scatter',
                 fontsize=13, fontweight='bold')

    # A. Diagrama de cajas: longitud igualada al segmento más corto para comparación justa
    n_min = min(len(data_before), len(data_seizure), len(data_after))
    axes[0].boxplot(
        [data_before[:n_min], data_seizure[:n_min], data_after[:n_min]],
        labels=['Antes', 'Crisis', 'Después']
    )
    axes[0].set_title(f'Diagrama de Cajas — Canal {canal}', fontweight='bold')
    axes[0].set_ylabel('Amplitud (μV)')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    # B. Histograma superpuesto de los 3 bloques + ajuste de PDF sobre la crisis
    for datos, label, color in [
        (data_before,  'Before',  'steelblue'),
        (data_seizure, 'Crisis',  'tomato'),
        (data_after,   'Después', 'seagreen'),
    ]:
        axes[1].hist(datos, bins=50, density=True, alpha=0.35, color=color, label=label)

    bins_range = np.linspace(data_seizure.min(), data_seizure.max(), 200)
    mu_n, std_n        = norm.fit(data_seizure)
    df_t, loc_t, scale_t = t.fit(data_seizure)
    axes[1].plot(bins_range, norm.pdf(bins_range, mu_n, std_n),
                 'k--', linewidth=2, label=f'Normal (μ={mu_n:.1f}, σ={std_n:.1f})')
    axes[1].plot(bins_range, t.pdf(bins_range, df_t, loc_t, scale_t),
                 'b-',  linewidth=2, label=f't-loc-scale (df={df_t:.1f})')
    axes[1].set_title(f'Histograma + PDF — Canal {canal}', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle=':', alpha=0.4)

    # C. Scatter (μ, σ) para TODOS los canales en los 3 estados
    for name, color, label in [
        ('before',  'steelblue', 'Before'),
        ('seizure', 'tomato',    'Crisis'),
        ('after',   'seagreen',  'After'),
    ]:
        axes[2].scatter(stats[name]['mean'], stats[name]['std'],
                        color=color, label=label, alpha=0.7, s=40)
    axes[2].set_title('Scatter μ vs σ — Todos los Canales', fontweight='bold')
    axes[2].set_xlabel('Media (μ)')
    axes[2].set_ylabel('Desviación Estándar (σ)')
    axes[2].legend()
    axes[2].grid(linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

# Calcula el mejor descriptor y umbral sin plots (importable por arch2.py).
BEST_DESCRIPTOR, UMBRAL_MIN_E1, UMBRAL_MAX_E1 = _best_descriptor()

if __name__ == '__main__':
    # 1. Análisis de discriminabilidad completo (con gráficos)
    analisis_discriminabilidad()

    # 2. Umbral dinámico por canal con el mejor descriptor
    escenario1paso2(BEST_DESCRIPTOR, UMBRAL_MIN_E1, UMBRAL_MAX_E1)

    # 3a. Heatmap de correlación de Pearson (sincronía normalizada)
    heatmap_pearson()

    # 3b. Heatmap de covarianza (sincronía en magnitud)
    heatmap_covarianza()

    # 4. Autocorrelación para todos los canales
    autocorr_todos_canales()

    # 5. Histograma, PDF, Boxplot y Scatter
    escenario1paso4()
