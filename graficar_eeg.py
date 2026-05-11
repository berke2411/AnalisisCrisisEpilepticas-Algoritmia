import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyedflib import highlevel

# --- Parámetros ---
EDF_FILE   = './archivos/chb20_12.edf'
START_SEC  = 94    # Inicio de la crisis (segundos)
END_SEC    = 123   # Fin de la crisis (segundos)
FS         = 256   # Frecuencia de muestreo (Hz)
CONTEXTO   = 30    # Segundos antes y después de la crisis

# --- Carga y recorte de la ventana de interés ---
signals, signal_headers, header = highlevel.read_edf(EDF_FILE)
n_canales, n_muestras = signals.shape
duracion_total = n_muestras / FS

t_ini = max(0, START_SEC - CONTEXTO)
t_fin = min(duracion_total, END_SEC + CONTEXTO)
s_ini = int(t_ini * FS)
s_fin = int(t_fin * FS)

tiempo = np.arange(s_ini, s_fin) / FS   # eje X en segundos absolutos

nombres = [h['label'].strip() for h in signal_headers]

# Filtrar canales sin señal (nombre "." o STD == 0 en el segmento)
canales_validos = [
    (nombre, signals[i, s_ini:s_fin])
    for i, nombre in enumerate(nombres)
    if nombre != '.' and signals[i, s_ini:s_fin].std() > 0
]
n_validos = len(canales_validos)

print(f"Ventana      : {t_ini:.0f}s  a  {t_fin:.0f}s  ({t_fin - t_ini:.0f}s)")
print(f"Crisis       : {START_SEC}s a {END_SEC}s  ({END_SEC - START_SEC}s)")
print(f"Canales      : {n_validos} (de {n_canales} totales, excluidos los vacios)")

# ─────────────────────────────────────────────────────────────────────────────
# Figura única — ventana ±30 s con alta amplitud
# ─────────────────────────────────────────────────────────────────────────────
SEPARACION = 1.0   # espacio entre líneas base de cada canal
AMPLITUD   = 0.42  # fracción de SEPARACION que ocupa cada señal (0.42 = ~84% del espacio)

altura_fig = max(14, n_validos * 0.65)
fig, ax = plt.subplots(figsize=(20, altura_fig))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

y_ticks  = []
y_labels = []

for i, (nombre, seg) in enumerate(canales_validos):
    std = seg.std()
    seg_norm = (seg - seg.mean()) / std * AMPLITUD   # centrado + escalado
    offset = (n_validos - 1 - i) * SEPARACION
    ax.plot(tiempo, seg_norm + offset,
            color='#58a6ff', linewidth=0.7, alpha=0.92)
    y_ticks.append(offset)
    y_labels.append(nombre)

# Región de crisis
ax.axvspan(START_SEC, END_SEC, alpha=0.15, color='#ff4444', zorder=0)
ax.axvline(START_SEC, color='#ff4444', linewidth=1.8, linestyle='--')
ax.axvline(END_SEC,   color='#ffaa00', linewidth=1.8, linestyle='--')

# Etiquetas de tiempo sobre las líneas verticales
ymax_label = (n_validos - 0.15) * SEPARACION
ax.text(START_SEC, ymax_label, f'  Inicio\n  {START_SEC}s',
        color='#ff6666', fontsize=8.5, va='top', fontweight='bold')
ax.text(END_SEC,   ymax_label, f'  Fin\n  {END_SEC}s',
        color='#ffcc44', fontsize=8.5, va='top', fontweight='bold')

# Ejes y etiquetas
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=8, color='#c9d1d9')
ax.set_xlabel('Tiempo (s)', color='#c9d1d9', fontsize=12)
ax.set_title(
    f'EEG — chb20_12.edf — {n_validos} canales\n'
    f'Ventana: {t_ini:.0f}s – {t_fin:.0f}s  |  Crisis: {START_SEC}s – {END_SEC}s  ({END_SEC - START_SEC}s)',
    color='white', fontsize=13, pad=12
)
ax.tick_params(axis='x', colors='#8b949e', labelsize=9)
ax.tick_params(axis='y', colors='#8b949e', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

# Leyenda
parche_crisis = mpatches.Patch(color='#ff4444', alpha=0.5, label=f'Crisis ({START_SEC}–{END_SEC}s)')
linea_inicio  = plt.Line2D([0], [0], color='#ff4444', linewidth=1.8, linestyle='--', label=f'Inicio ({START_SEC}s)')
linea_fin     = plt.Line2D([0], [0], color='#ffaa00', linewidth=1.8, linestyle='--', label=f'Fin ({END_SEC}s)')
ax.legend(handles=[parche_crisis, linea_inicio, linea_fin],
          loc='upper right', fontsize=9,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

ax.set_xlim(t_ini, t_fin)
ax.set_ylim(-SEPARACION * 0.6, n_validos * SEPARACION)

# Líneas de cuadrícula verticales cada 10 segundos
for t_grid in range(int(t_ini), int(t_fin) + 1, 10):
    ax.axvline(t_grid, color='#30363d', linewidth=0.5, zorder=0)

plt.tight_layout()
plt.savefig('eeg_crisis.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Guardado: eeg_crisis.png")
plt.show()
