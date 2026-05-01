import numpy as np
from scipy import signal
from pyedflib import highlevel
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# --- Parámetros ---
# Ruta del archivo de electroencefalograma (EEG) en formato EDF.
EDF_FILE = './archivos/chb20_12.edf' 
# Tiempos marcados en segundos donde inicia y termina la crisis epiléptica (seizure).
START_SEC = 94
END_SEC = 123
# Tamaño de la ventana de contexto (en segundos) que queremos observar antes y después de la crisis.
WINDOW_SEC = 120

# --- Carga ---
# Lee el archivo EDF. Extrae las señales (datos crudos), la información de cada canal y el encabezado general.
signals, signal_headers, header = highlevel.read_edf(EDF_FILE)
# Frecuencia de muestreo (muestras por segundo). Aquí está hardcodeada a 256 Hz.
fs = 256 

# --- Muestras ---
# Convierte los tiempos (segundos) a índices de array (muestras) multiplicando por la frecuencia de muestreo.
start_sample = START_SEC * fs
end_sample = END_SEC * fs
window_samples = WINDOW_SEC * fs
total_samples = signals.shape[1] # Número total de muestras en la señal

# Calcula dónde empiezan y terminan las ventanas de contexto.
# Usa max(0, ...) para evitar índices negativos si la crisis ocurre muy cerca del inicio.
inicio_before = max(0, start_sample - window_samples)
# Usa min(total, ...) para evitar salirnos del límite del archivo si la crisis ocurre cerca del final.
fin_after = min(total_samples, end_sample + window_samples)

# --- Segmentación y centrado ---
def extract_and_center(sig, s, e):
    """
    Extrae un fragmento de la señal y la "centra" restándole la media a cada canal.
    Esto elimina el componente de corriente continua (DC offset) o línea base.
    """
    segment = sig[:, s:e] # Extrae todas las filas (canales) y las columnas desde 's' hasta 'e'
    # Resta la media de cada canal (axis=1). keepdims=True mantiene la forma para poder restar correctamente.
    return segment - segment.mean(axis=1, keepdims=True)

# Extraemos los tres bloques de interés ya centrados en cero:
before_centered = extract_and_center(signals, inicio_before, start_sample) # Antes de la crisis
seizure_centered = extract_and_center(signals, start_sample, end_sample)   # Durante la crisis
after_centered   = extract_and_center(signals, end_sample, fin_after)      # Después de la crisis

# Une los tres segmentos consecutivamente en un solo bloque grande para análisis global.
total_block = np.concatenate(
    (before_centered, seizure_centered, after_centered), axis=1
)

# imprimo resultados iniciales
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Dimensiones del bloque 'Before': {before_centered.shape}")
print(f"Dimensiones del bloque 'Crisis': {seizure_centered.shape}")
print(f"Dimensiones del bloque 'After': {after_centered.shape}")
print(f"Dimensiones del 'Bloque Total': {total_block.shape}")


# --- Descriptores ---
def compute_stats(seg):
    """
    Calcula diversas métricas estadísticas para un segmento de señal dado.
    """
    var = np.var(seg, axis=1) # Varianza: qué tan dispersos están los datos
    return {
        'var':      var,
        'mean':     np.mean(seg,axis=1),
        'std':      np.sqrt(var), # Desviación estándar: raíz de la varianza
        'abs_mean': np.mean(np.abs(seg), axis=1), # Media de los valores absolutos (amplitud promedio)
        'cov':      np.cov(seg), # Matriz de covarianza: cómo varían los canales juntos
        'pearson':  np.corrcoef(seg), # Matriz de correlación de Pearson: relación lineal entre canales (-1 a 1)
    }

# Agrupamos los segmentos en un diccionario para procesarlos fácilmente con un bucle.
segments = {
    'before':  before_centered,
    'seizure': seizure_centered,
    'after':   after_centered,
}

# Creamos un diccionario 'stats' que calcula y guarda las métricas para cada uno de los 3 segmentos.
stats = {name: compute_stats(seg) for name, seg in segments.items()}

# --- Autocorrelación ---
C0 = 0 # Índice del canal a analizar (Canal 0)
# Calcula la autocorrelación para el canal 0 en los tres segmentos.
# Esto mide qué tan similar es la señal consigo misma al desplazarla en el tiempo.
autocorr = {
    name: signal.correlate(seg[C0], seg[C0], mode='full')
    for name, seg in segments.items()
}

# --- Reporte ---
C0, C1 = 0, 1 # Seleccionamos los canales 0 y 1 para mostrar el reporte en pantalla

print(f"fs: {fs} Hz | Bloque total: {total_block.shape}")

# Imprime la varianza, desviación estándar y media absoluta SOLO para el canal 0 (C0) en los 3 periodos.
for metric in ('var', 'std', 'abs_mean'):
    vals = " | ".join(
        f"{name}: {stats[name][metric][C0]:.2f}"
        for name in ('before', 'seizure', 'after')
    )
    print(f"{metric.upper():12} -> {vals}")

# Imprime la covarianza y la correlación entre el canal 0 (C0) y el canal 1 (C1)
# Compara el estado "antes" (before) con el estado "durante" (seizure).
print(f"\nCovarianza C{C0}-C{C1}  -> "
      f"Antes: {stats['before']['cov'][C0, C1]:.2f} | "
      f"Crisis: {stats['seizure']['cov'][C0, C1]:.2f}")
print(f"Pearson C{C0}-C{C1}     -> "
      f"Antes: {stats['before']['pearson'][C0, C1]:.2f} | "
      f"Crisis: {stats['seizure']['pearson'][C0, C1]:.2f}")


# --- ESCENARIO 1: Umbral Dinámico [min, max] de un Descriptor Estadístico ---
def escenario1paso2():
    descriptor = 'std'  # Podés cambiar a 'var','std', 'abs_mean', etc.

    # Extraemos el descriptor para todos los canales en cada bloque
    vals_before  = stats['before'][descriptor]   # shape: (n_canales,)
    vals_seizure = stats['seizure'][descriptor]
    vals_after   = stats['after'][descriptor]

    # Umbral dinámico: [min, max] del descriptor en el bloque 'before'
    umbral_min = vals_before.min()
    umbral_max = vals_before.max()

    n_canales = len(vals_before)
    x_pos = np.arange(n_canales)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)
    bloques = [
        ('Antes (Reposo)',        vals_before,  'steelblue'),
        ('Crisis (Epilepsia)',    vals_seizure, 'tomato'),
        ('Después (Recuperación)',vals_after,   'seagreen'),
    ]

    for ax, (titulo, vals, color) in zip(axes, bloques):
        # Zona umbral (rango normal definido por 'before')
        ax.axhspan(umbral_min, umbral_max, color='gold', alpha=0.25, zorder=0,
                label=f'Rango Normal [{umbral_min:.2f}, {umbral_max:.2f}]')
        ax.axhline(umbral_max, color='orange', linestyle='--', linewidth=1.2)
        ax.axhline(umbral_min, color='orange', linestyle='--', linewidth=1.2)

        # Coloreamos en rojo los canales que superan el umbral
        for i, v in enumerate(vals):
            fuera = v > umbral_max or v < umbral_min
            ax.bar(i, v, color='red' if fuera else color,
                alpha=0.8, width=0.7, zorder=2)

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


def escenario1paso4():
    canal = 0

    # 1. Extraemos las señales del Canal 0 para los 3 bloques
    data_before = before_centered[canal, :]
    data_seizure = seizure_centered[canal, :]
    data_after = after_centered[canal, :]

    # Creamos una figura con 3 subgráficos horizontales. 'axes' es un arreglo de 3 posiciones
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- A. DIAGRAMA DE CAJAS ---
    n_min = min(len(data_before), len(data_seizure), len(data_after)) 
                # Lo usamos ya que la crisis es mucho mas corta que el resto de bloques, haciendo que el grafico no sea tan representativo, ...
                #...  de esta manera el tamaño muestreal es igual para las 3 (evitando excesivos outliers)
    
    axes[0].boxplot(
        [data_before[:n_min], data_seizure[:n_min], data_after[:n_min]],
        labels=['Antes', 'Crisis', 'Después']
    )
    axes[0].set_title(f'Diagrama de Cajas - Canal {canal}', fontweight='bold')
    axes[0].set_ylabel('Amplitud de la señal (μV)')
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)

    # --- B. HISTOGRAMA Y PDF en axes[1] ---
    # Graficamos el histograma del bloque 'Crisis' como ejemplo
    counts, bins, patches = axes[1].hist(data_seizure, bins=50, density=True, alpha=0.5, color='tomato', label='Datos empíricos (Crisis)')

    # Ajustamos una PDF Normal (Gaussiana)
    mu, std = norm.fit(data_seizure)
    p_norm = norm.pdf(bins, mu, std)
    axes[1].plot(bins, p_norm, 'k--', linewidth=2, label=f'Normal\n(μ={mu:.2f}, σ={std:.2f})')

    # Ajustamos una PDF t-location-scale (t-Student) sugerida en la teoría
    df, loc, scale = t.fit(data_seizure)
    p_t = t.pdf(bins, df, loc, scale)
    axes[1].plot(bins, p_t, 'b-', linewidth=2, label='t-location-scale')

    axes[1].set_title('Histograma y Ajuste de PDF (Crisis)', fontweight='bold')
    axes[1].legend()

    # --- C. SCATTER PLOT (Diagrama de dispersión de parámetros) en axes[2] ---
    # Extraemos la media y desviación estándar de todos los 28 canales usando tu diccionario 'stats'
    mu_before = stats['before']['mean']
    sigma_before = stats['before']['std']

    mu_seizure = stats['seizure']['mean']
    sigma_seizure = stats['seizure']['std']

    axes[2].scatter(mu_before, sigma_before, color='steelblue', label='0 (Normal)', alpha=0.7)
    axes[2].scatter(mu_seizure, sigma_seizure, color='tomato', label='1 (Crisis)', alpha=0.7)
    axes[2].set_title('Scatter plot classification (μ vs σ)', fontweight='bold')
    axes[2].set_xlabel('Media (μ)')
    axes[2].set_ylabel('Desviación Estándar (σ)')
    axes[2].legend()
    axes[2].grid(linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    
#escenario1paso2()
escenario1paso4()

