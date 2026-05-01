import numpy as np
from scipy import signal
from pyedflib import highlevel

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