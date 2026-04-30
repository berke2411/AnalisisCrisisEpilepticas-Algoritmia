# pip install pyedflib
import numpy as np
from pyedflib import highlevel

# 1. Definir los parámetros extraídos del archivo summary.txt del paciente
edf_file = './archivos/chb20_12.edf'  # Archivo de ejemplo
start_sec = 94           # Seizure Start Time en segundos
end_sec = 123             # Seizure End Time en segundos

# 2. Cargar las señales y metadatos usando highlevel
signals, signal_headers, header = highlevel.read_edf(edf_file)

# Extraemos la frecuencia de muestreo (fs) del primer canal (suele ser igual para todos)
#print(signal_headers)
fs = 256

# 3. Conversión de tiempos (Regla de 3 simple: 1 Hz = 1 muestra por segundo)
# Calculamos en qué "muestra" exacta empieza y termina la crisis
start_sample = start_sec * fs
end_sample = end_sec * fs

# Definimos el tamaño de la ventana requerida: 2 minutos = 120 segundos
window_samples = 120 * fs

# 4. Segmentación - Escenario 1 (Por Bloques)
# Señal total = 2 min antes + crisis + 2 min después
# Usamos max() para evitar índices negativos si la crisis empieza antes de los 2 min

# Obtenemos la cantidad total de muestras que tiene el archivo (columnas de la matriz)
total_samples = signals.shape[1]


# Límite inferior: usamos max() para no tener índices negativos
inicio_before = max(0, start_sample - window_samples)

# Límite superior: usamos min() para no pasarnos del largo total del archivo
fin_after = min(total_samples, end_sample + window_samples)

before_seizure = signals[:, inicio_before : start_sample]
seizure = signals[:, start_sample : end_sample]
after_seizure = signals[:, end_sample : fin_after]

# 5. Centrado de señales
# Fundamental: Restarle la media a cada segmento por canal para que se centre.
# Usamos axis=1 para calcular la media de cada fila (canal) y keepdims para mantener la dimensión.
before_centered = before_seizure - np.mean(before_seizure, axis=1, keepdims=True)
seizure_centered = seizure - np.mean(seizure, axis=1, keepdims=True)
after_centered = after_seizure - np.mean(after_seizure, axis=1, keepdims=True)

# 6. Escenario 2 (Bloque Total)
# Unimos los bloques ya centrados para tener la señal continua: [Before, Crisis, After]
total_block = np.concatenate((before_centered, seizure_centered, after_centered), axis=1)

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Dimensiones del bloque 'Before': {before_centered.shape}")
print(f"Dimensiones del bloque 'Crisis': {seizure_centered.shape}")
print(f"Dimensiones del bloque 'After': {after_centered.shape}")
print(f"Dimensiones del 'Bloque Total': {total_block.shape}")