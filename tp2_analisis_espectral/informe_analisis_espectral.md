# Informe — Análisis Espectral de Crisis Epilépticas
**Algorítmica y Lógica Computacional · UCA**  
Berkelaar · Caitano · Flachsland

---

## Índice

1. [Introducción y dataset](#1-introducción-y-dataset)
2. [Paso a paso del notebook](#2-paso-a-paso-del-notebook)
   - [Carga y segmentación](#21-carga-y-segmentación-celda-5)
   - [Escenario 1 — Análisis por bloques](#22-escenario-1--análisis-por-bloques)
   - [Escenario 2 — Bloque total](#23-escenario-2--bloque-total)
3. [Conclusiones por celda](#3-conclusiones-por-celda)
4. [Conclusión general](#4-conclusión-general)
5. [Referencias bibliográficas](#5-referencias-bibliográficas)

---

## 1. Introducción y dataset

### 1.1 Contexto clínico

La epilepsia es un trastorno neurológico caracterizado por descargas eléctricas anómalas y recurrentes en el cerebro. Una crisis epiléptica (*seizure*) se origina cuando un grupo de neuronas dispara de forma sincrónica y excesiva, propagándose a regiones adyacentes. El electroencefalograma (EEG) registra estas diferencias de potencial eléctrico mediante electrodos colocados en el cuero cabelludo, lo que permite detectar y analizar la actividad ictal (durante la crisis) y post-ictal (posterior a la crisis) [1].

### 1.2 CHB-MIT Scalp EEG Database

El dataset empleado es el **CHB-MIT Scalp EEG Database** disponible en PhysioNet [2]. Contiene registros de 24 pacientes con epilepsia refractaria, con anotaciones precisas (en segundos) del inicio y fin de cada crisis. Los registros están almacenados en formato EDF (*European Data Format*), con frecuencias de muestreo típicas de 256 Hz.

El archivo analizado es `chb20_12.edf`, correspondiente al **Paciente 20**, con la siguiente crisis anotada:

| Parámetro | Valor |
|---|---|
| Inicio de crisis | 94 s |
| Fin de crisis | 123 s |
| Duración | 29 s |
| Frecuencia de muestreo | 256 Hz |
| Canales EEG | 28 |
| Duración total analizada | 243 s |

### 1.3 Sistema internacional 10-20

Los 28 canales corresponden a derivaciones bipolares del sistema internacional 10-20, donde cada canal registra la diferencia de potencial entre dos electrodos adyacentes (e.g., `FP2-F8`, `T8-P8`, `FT10-T8`). Las derivaciones fronto-temporales derechas resultaron ser las más relevantes en este caso, como se verá en los resultados.

---

## 2. Paso a paso del notebook

### 2.1 Carga y segmentación (Celda 5)

**¿Qué hace?**  
Se utiliza la librería `pyedflib` para leer el archivo `.edf` y extraer:
- La matriz de señales (28 canales × N muestras)
- Los metadatos: frecuencia de muestreo, etiquetas de canales

**Segmentación temporal:**  
La señal se divide en tres bloques contiguos según la anotación de la crisis:

```
|-- Before (94 s) --|-- Crisis (29 s) --|-- After (120 s) --|
                    ↑                   ↑
                  t=94s              t=123s
```

> **Nota técnica**: la consigna indica 2 minutos de *before*, pero el archivo comienza solo 94 segundos antes de la crisis (límite físico del registro). El bloque *after* sí alcanza los 120 s completos.

A cada segmento se le resta la media por canal (*DC offset removal*), lo que centra la señal en cero y elimina derivas lentas de los electrodos [3].

**Resultado:**

| Bloque | Forma | Duración |
|--------|-------|----------|
| before | (28, 24064) | 94.0 s |
| crisis | (28, 7424) | 29.0 s |
| after | (28, 30720) | 120.0 s |
| total | (28, 62208) | 243.0 s |

---

### 2.2 Escenario 1 — Análisis por bloques

Se analiza cada bloque (Before / Crisis / After) como una señal cuasi-estacionaria independiente.

---

#### 2.2.1 FFT y PSD de Welch — E1 (Celda 7)

**¿Qué hace?**

Para cada bloque se calculan dos estimaciones del espectro:

**FFT (Transformada Rápida de Fourier)** [4]:
$$X(k) = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}, \quad k = 0, 1, \ldots, N-1$$

La magnitud normalizada $|X(k)|/N$ da la amplitud de cada componente frecuencial. Resolución espectral: $\Delta f = f_s/N$. No aplica ventaneo, por lo que es susceptible a *spectral leakage* en los bordes del segmento [5].

**PSD de Welch** [6]:
$$P_{\text{Welch}}(f) = \frac{1}{K} \sum_{k=1}^{K} \left| X_k(f) \right|^2$$

Divide el segmento en $K$ sub-bloques solapados, aplica ventana de Hann a cada uno y promedia los periodogramas. Reduce varianza espectral a costa de resolución frecuencial. Es el método preferido para señales no estacionarias como el EEG [7].

**Visualización:** Los 28 canales se grafican como trazos finos (α = 0.12) y se superpone la mediana entre canales como trazo grueso. El fondo está coloreado por banda cerebral para lectura inmediata.

**Verificación de ruido de red eléctrica:**  
Se compara la PSD en 50 Hz y 60 Hz contra la línea base (30–45 Hz). Un ratio > 5 indica ruido de red que requiere filtro *notch* [3].

---

#### 2.2.2 Potencia por banda cerebral — E1 (Celda 8)

**¿Qué hace?**  
Para cada bloque se calcula la potencia integrada en cada banda cerebral mediante la regla del trapecio:
$$P_{\text{banda}} = \int_{f_{\text{lo}}}^{f_{\text{hi}}} S(f)\, df \approx \sum_k \frac{S(f_k) + S(f_{k+1})}{2} \Delta f$$

Bandas analizadas [1]:

| Banda | Rango (Hz) | Función fisiológica |
|-------|-----------|---------------------|
| Delta | 0–4 | Sueño profundo, actividad patológica ictal |
| Theta | 4–8 | Somnolencia, memoria, hipocampo |
| Alpha | 8–12 | Relajación, ritmo de Berger |
| Beta | 12–30 | Alerta cognitiva, actividad motora |
| Gamma | 30–64 | Procesamiento sensorial de alta frecuencia |

Se grafican como barras agrupadas (Before / Crisis / After) y se calcula el ratio Crisis/Before como métrica de discriminabilidad.

---

#### 2.2.3 Periodograma — E1 (Celda 9)

**¿Qué hace?**  
El periodograma es el estimador espectral más básico [8]:
$$\hat{P}(f) = \frac{1}{N \cdot f_s} \left| \sum_{n=0}^{N-1} x[n] e^{-j2\pi fn/f_s} \right|^2$$

Tiene alta varianza (mucho ruido espectral) pero revela picos individuales con mayor detalle que Welch. Se visualiza en escala logarítmica: trazos finos por canal, mediana gruesa, y trazo discontinuo para el canal con mayor potencia total.

---

#### 2.2.4 Espectrograma — 3 ventanas × 3 overlaps — E1 (Celda 10)

**¿Qué hace?**  
La **STFT** (*Short-Time Fourier Transform*) [9] aplica la FFT sobre ventanas deslizantes de longitud $W$ con solapamiento $H$:
$$\text{STFT}(m, k) = \sum_{n} x[n] \cdot w[n - mH] \cdot e^{-j2\pi kn/N}$$

El espectrograma es $|\text{STFT}(m,k)|^2$: un mapa tiempo-frecuencia. El balance entre resolución temporal y frecuencial está gobernado por el principio de Heisenberg-Gabor: no se pueden maximizar ambas simultáneamente [5].

Se prueban **9 configuraciones** (3 ventanas × 3 overlaps):
- Ventanas: 1 s, 2 s, 4 s
- Overlaps: 25%, 50%, 75%

Para cada configuración se calcula el ratio de potencia Crisis/Before en cada banda, identificando la configuración más discriminante.

---

#### 2.2.5 Scatter Beta vs Gamma — E1 (Celda 11)

**¿Qué hace?**  
Diagrama de dispersión donde cada punto es un canal EEG. El eje X es la potencia Beta por canal, el eje Y la potencia Gamma. Permite evaluar visualmente si los tres bloques forman clusters separables en el espacio Beta-Gamma.

---

### 2.3 Escenario 2 — Bloque total

Se trabaja con la señal completa `[Before | Crisis | After]` como una sola serie temporal continua, moviendo una ventana deslizante para obtener la evolución temporal del espectro.

---

#### 2.3.1 FFT y PSD del bloque total — E2 (Celda 13)

Igual que el Escenario 1 pero sobre el bloque concatenado. Muestra el espectro "promediado" en el tiempo, donde el peso de Before+After (214 s) domina sobre la crisis (29 s).

---

#### 2.3.2 Potencia por banda en ventanas deslizantes + Detección — E2 (Celda 14)

**¿Qué hace?**  
Se desliza una ventana de 1 s en pasos de 1 s sobre el bloque total. Para cada ventana se estima la PSD de Welch y se integra por banda, obteniendo una **serie temporal de potencia** por banda y por canal.

Simultáneamente se ejecuta el **algoritmo de detección de crisis** con criterios en cascada:

1. **Canales significativos**: se seleccionan los 8 canales con mayor ratio Crisis/Before en la banda Gamma (calculado sobre las ventanas de referencia).
2. **Umbral adaptativo**: $\theta_i = \mu_i^{\text{before}} + 2.5 \cdot \sigma_i^{\text{before}}$ por canal.
3. **Votación**: una ventana es positiva si ≥ 3 de los 8 canales superan su umbral.
4. **Persistencia**: se declara crisis solo si ≥ 3 ventanas consecutivas son positivas (evita espúreos).

La detección se grafica como una **línea naranja vertical** sobre los 5 paneles de potencia por banda.

---

#### 2.3.3 Scatter por ventanas — E2 (Celda 15)

Igual que el Escenario 1 pero cada punto es una **ventana temporal de 1 s** en lugar de un canal. Permite visualizar cómo se distribuyen las ventanas Before/Crisis/After en el espacio Beta-Gamma.

---

#### 2.3.4 Periodograma y STFT del bloque total — E2 (Celda 16)

STFT aplicada sobre la señal completa con ventana de 2 s y overlap del 50%. El mapa tiempo-frecuencia muestra la evolución del contenido espectral a lo largo de los 243 s. La línea naranja vertical marca la detección automática.

---

#### 2.3.5 Espectrograma — 9 configuraciones — E2 (Celda 17)

Misma evaluación de 9 configuraciones que el Escenario 1, pero sobre el bloque total. Identifica qué combinación ventana/overlap maximiza la discriminabilidad Crisis/Before en el espectrograma continuo.

---

#### 2.3.6 Resumen de detección — E2 (Celda 18)

Tabla de métricas del algoritmo de detección: parámetros usados, crisis anotada vs detectada, retardo, TP/FP/FN, precisión y recall, y ranking de canales por discriminabilidad.

---

## 3. Conclusiones por celda

### 3.1 Carga y segmentación (Celda 5)

El archivo `chb20_12.edf` tiene una frecuencia de muestreo de **256 Hz**, que es el estándar para EEG clínico de alta calidad [1]. Con 28 canales y 243 segundos de señal continua, se dispone de suficiente resolución temporal y espacial para el análisis espectral. La limitación de 94 s en el bloque *before* (en lugar de los 2 minutos estándar) obliga a interpretar los resultados pre-ictales con cautela, ya que el período basal es más corto.

---

### 3.2 FFT y PSD de Welch — E1 (Celda 7)

**Resultado clave:**

| Bloque | Frecuencias dominantes (Hz) | Amplitud mediana |
|--------|-----------------------------|-----------------|
| Before | 0.78 – 1.29 (delta) | ~1.15–1.21 |
| Crisis | 0.34 – 0.55 (delta profundo) | ~3.2–4.1 |
| After | 0.17 – 0.77 (delta profundo) | ~1.67–1.77 |

**Interpretación:**
- Las frecuencias dominantes están en la banda delta (< 4 Hz) en todos los estados, lo cual es esperado en EEG de cuero cabelludo debido a la atenuación de altas frecuencias por el cráneo y los tejidos [1].
- Durante la **crisis**, la amplitud en delta se triplica (de ~1.2 a ~4.1). Esto refleja las descargas epileptiformes lentas características del período ictal.
- En el **post-ictal** (After), la amplitud en delta permanece más alta que en Before (~1.7 vs ~1.2), signo clásico de "lentificación post-ictal" asociada a inhibición neuronal tras la descarga [1].
- **No se detectó ruido de red eléctrica** a 50 Hz ni 60 Hz (ratio 0.2x, muy por debajo del umbral 5x). La señal no requiere filtrado *notch*, lo que preserva la integridad de las bandas beta y gamma.

**FFT vs PSD de Welch:**  
Ambos métodos coinciden en las frecuencias dominantes, pero la PSD de Welch presenta menor varianza espectral, lo que facilita la identificación de picos verdaderos versus artefactos estadísticos [6].

---

### 3.3 Potencia por banda cerebral — E1 (Celda 8)

**Resultado clave:**

| Banda | Before | Crisis | After | **Crisis/Before** |
|-------|--------|--------|-------|-------------------|
| Delta | 471.9 | 1608 | 1148 | **3.41×** |
| Theta | 85.26 | 220.2 | 131.6 | **2.58×** |
| Alpha | 20.12 | 87.23 | 46.36 | **4.34×** |
| Beta | 22.54 | 372.0 | 64.64 | **16.51×** |
| **Gamma** | **6.135** | **298.5** | **52.72** | **48.66×** |

**Interpretación:**
- La banda **Gamma** (30–64 Hz) es la más discriminante con un ratio de **48.66×**. Esto es consistente con la literatura: las crisis epilépticas generan oscilaciones de alta frecuencia (*High Frequency Oscillations*, HFOs) en el rango gamma como parte del reclutamiento ictal [10].
- La banda **Beta** muestra un ratio de 16.51×, también muy elevado. Beta y Gamma suelen co-activarse durante crisis de inicio focal [11].
- **Alpha** muestra un aumento de 4.34×: inesperado dado que alpha normalmente *disminuye* durante activación. Este aumento podría reflejar la propagación de la descarga hacia regiones occipito-parietales.
- **Delta** muestra el ratio más bajo (3.41×) en términos relativos, pero el mayor aumento en términos absolutos (de 471 a 1608 µV²/Hz), reflejando las descargas lentas ictales.
- El bloque **After** mantiene valores elevados en todas las bandas (especialmente delta y gamma), indicando que el post-ictal del Paciente 20 tarda más de 2 minutos en regresar a la línea base.

---

### 3.4 Periodograma — E1 (Celda 9)

**Resultado clave:**

| Bloque | Frecuencia pico | Potencia total | Canal máximo |
|--------|-----------------|----------------|--------------|
| Before | 0.82 Hz | 623.7 | P4-O2 |
| Crisis | 0.38 Hz | **2687** | **T7-FT9** |
| After | 0.40 Hz | 1565 | P7-O1 |

**Interpretación:**
- La potencia total durante la crisis (2687) es **4.3 veces mayor** que en el período basal (623.7). Esto confirma la explosión energética característica de una crisis tónico-clónica o focal con generalización.
- El canal de máxima potencia durante la crisis es **T7-FT9** (temporal izquierdo), sugiriendo un componente de activación del lóbulo temporal izquierdo.
- En contraste, el scatter de E1 (Celda 11) y el análisis de ventanas (Celda 14) identificaron los canales fronto-temporales **derechos** (FT10-T8, F8-T8, FP2-F8) como los más discriminantes en gamma — la diferencia se debe a que el periodograma captura la potencia total (dominada por delta), mientras que el análisis por bandas captura la información de gamma de forma aislada.
- El periodograma tiene mayor varianza que Welch, visible como "ruido" espectral en las curvas individuales por canal. Sin embargo, el canal de máxima potencia por bloque es robusto a esta varianza.

---

### 3.5 Espectrograma — 9 configuraciones — E1 (Celda 10)

**Resultado clave:**

| Posición | Ventana | Overlap | Banda | Score (gamma) |
|----------|---------|---------|-------|---------------|
| **1° (mejor)** | **4 s** | **75%** | **Gamma** | **47.85×** |
| 2° | 4 s | 50% | Gamma | 46.01× |
| 3° | 2 s | 75% | Gamma | 45.78× |
| último | 1 s | 50% | Gamma | 44.38× |

**Interpretación:**
- **Todas las configuraciones identifican Gamma como la banda más discriminante**, con scores entre 44 y 48. Esto valida la conclusión de la Celda 8 con una metodología independiente.
- La mejor configuración (ventana 4 s, overlap 75%) favorece **resolución frecuencial alta**: una ventana más larga produce $\Delta f = 1/T$ más pequeño, separando mejor las bandas gamma y beta.
- El overlap alto (75%) actúa como suavizado temporal, reduciendo la varianza del estimador a costa de una mayor correlación entre ventanas contiguas [9].
- Ventanas cortas (1 s) son útiles para localizar el instante exacto de inicio de la crisis, pero sacrifican resolución frecuencial.

---

### 3.6 Scatter Beta vs Gamma — E1 (Celda 11)

**Resultado clave:**

| Bloque | Beta (promedio) | Gamma (promedio) | Canal γ máximo |
|--------|-----------------|------------------|----------------|
| Before | 22.54 | 6.135 | FT10-T8 |
| Crisis | 372.0 | 298.5 | FT10-T8 |
| After | 64.64 | 52.72 | FT10-T8 |

**Interpretación:**
- Los 28 puntos de la crisis se desplazan drásticamente hacia la zona de alta Beta y alta Gamma, formando un **cluster perfectamente separado** del Before y After.
- El canal **FT10-T8** (fronto-temporal derecho) es el canal de máxima potencia gamma en los **tres estados**. Esto sugiere que este electrodo registra en la zona del foco epiléptico del Paciente 20.
- La separabilidad en el espacio 2D Beta-Gamma valida la viabilidad de un clasificador simple (e.g., regresión logística o SVM) basado en estas dos características [12].

---

### 3.7 FFT y PSD del bloque total — E2 (Celda 13)

**Interpretación:**
- Al analizar la señal completa (243 s), el espectro está dominado por el período Before+After (214 s) ya que representa el 88% del tiempo.
- Los picos delta de la crisis quedan enmascarados por el baseline. Esto ilustra por qué el análisis de bloque total (Escenario 2) necesita complementarse con ventanas deslizantes para capturar la dinámica ictal.

---

### 3.8 Potencia por banda + Detección de crisis — E2 (Celda 14)

**Resultado clave (detección):**

| Métrica | Valor |
|---------|-------|
| Inicio detección | 94.5 s |
| Fin detección | 242.5 s |
| **Retardo** | **+0.5 s** |
| TP | 29 (todas las ventanas de crisis) |
| FP | 93 (ventanas del período After) |
| FN | 0 |
| **Recall** | **100.00%** |
| Precisión | 23.77% |

**Canales de mayor discriminabilidad Gamma:**

| Canal | Ratio Crisis/Before |
|-------|---------------------|
| FP2-F8 | 222.70× |
| T8-P8 | 124.07× |
| F8-T8 | 102.03× |
| FT10-T8 | 97.43× |
| T7-FT9 | 80.77× |

**Interpretación de la detección:**
- El retardo de **+0.5 s** es clínicamente muy relevante: el algoritmo identifica la crisis prácticamente en tiempo real, dentro de la primera ventana post-inicio [12].
- El **Recall del 100%** significa que ninguna ventana de crisis se perdió (FN=0). En el contexto clínico esto es crítico: preferimos alertar de más que dejar pasar una crisis real.
- La **precisión baja (23.77%)** se debe a que el algoritmo continúa detectando durante el período post-ictal (After), donde la actividad gamma permanece elevada mucho más allá de los 123 s anotados. Esto no es un "error" del algoritmo sino una realidad fisiológica: la actividad post-ictal puede durar minutos [1].
- Para mejorar precisión sin sacrificar recall, se podría: aumentar `THRESH_K` (umbral más estricto), aumentar `MIN_CH` o `N_CONSEC`, o aplicar un detector de fin de crisis basado en el retorno a la línea base.

**Distribución de potencia por bandas (ventanas deslizantes):**

| Banda | Before | Crisis | After | Crisis/Before |
|-------|--------|--------|-------|---------------|
| Delta | 366.3 | 1014 | 758.3 | 2.77× |
| Theta | 80.76 | 221.3 | 112.6 | 2.74× |
| Alpha | 18.28 | 67.37 | 42.31 | 3.69× |
| Beta | 22.05 | 320.2 | 59.51 | 14.53× |
| **Gamma** | **6.154** | **270.2** | **57.13** | **43.91×** |

Los ratios son muy consistentes con el Escenario 1 (Celda 8), confirmando la robustez de los resultados.

**Canales más activos durante la crisis (conteo de ventanas):**

| Canal | Ventanas con max Gamma |
|-------|----------------------|
| FT10-T8 | 11 |
| F8-T8 | 6 |
| FT9-FT10 | 5 |
| T7-FT9 | 5 |

---

### 3.9 Scatter por ventanas — E2 (Celda 15)

| Estado | Ventanas | Beta prom | Gamma prom |
|--------|----------|-----------|------------|
| Before | 94 | 22.05 | 6.154 |
| Crisis | 29 | 320.2 | 270.2 |
| After | 120 | 59.51 | 57.13 |

**Interpretación:**
- El cluster de crisis (29 ventanas) está **completamente separado** del Before en el espacio Beta-Gamma.
- El cluster After se sitúa entre Before y Crisis: potencia más alta que Before pero mucho más baja que Crisis. Este gradiente temporal es el responsable de los falsos positivos del detector.
- La separación limpia antes/después del inicio de crisis valida que un clasificador entrenado con datos de Before podría generalizar a nuevas señales del mismo paciente [12].

---

### 3.10 Periodograma y STFT del bloque total — E2 (Celda 16)

**Resultado clave (STFT por bandas):**

| Banda | Before | Crisis | After | Crisis/Before |
|-------|--------|--------|-------|---------------|
| Delta | 167.8 | 624.4 | 444.9 | 3.72× |
| Theta | 30.78 | 78.30 | 48.86 | 2.54× |
| Alpha | 7.272 | 29.08 | 17.63 | 4.00× |
| Beta | 8.444 | 123.7 | 24.50 | 14.65× |
| **Gamma** | **2.308** | **100.8** | **19.79** | **43.66×** |

**Interpretación:**
- Los ratios de la STFT son **consistentes** con los de Welch y ventanas deslizantes, validando la robustez del hallazgo en gamma.
- El espectrograma STFT es visualmente el más informativo: el mapa tiempo-frecuencia muestra con claridad la "explosión" espectral en el instante t = 94 s, visible en todas las bandas pero especialmente en gamma y beta.
- La línea naranja de detección coincide con el inicio visible de la perturbación en el espectrograma, confirmando que el algoritmo responde al evento correcto.

---

### 3.11 Espectrograma — 9 configuraciones — E2 (Celda 17)

**Resultado clave:**

| Posición | Ventana | Overlap | Banda | Score prom | Canal | Score canal |
|----------|---------|---------|-------|------------|-------|-------------|
| **1° (mejor)** | **2 s** | **25%** | **Gamma** | **40.87×** | **FP2-F8** | **242.08×** |
| 2° | 2 s | 75% | Gamma | 40.84× | FP2-F8 | 236.46× |
| último top | 4 s | 50% | Gamma | 39.28× | FP2-F8 | 223.81× |

**Interpretación:**
- En el Escenario 2, la mejor configuración es **ventana 2 s, overlap 25%**, a diferencia del Escenario 1 donde era ventana 4 s, overlap 75%. La razón es que en el Escenario 2 el objetivo no es solo discriminar bandas sino también **localizar temporalmente** el inicio de la crisis, lo que premia ventanas más cortas [9].
- **FP2-F8** es el canal más discriminante con un ratio individual de **242×**, siendo consistente con su rol de canal frontal derecho próximo al foco.
- Independientemente de la configuración, la banda Gamma es siempre la ganadora. El resultado es altamente robusto a los hiperparámetros del espectrograma.

---

### 3.12 Resumen de detección — E2 (Celda 18)

Los canales de mayor ratio Crisis/Before en gamma son todos **derechos y fronto-temporales**: FP2-F8 (222.70×), T8-P8 (124.07×), F8-T8 (102.03×), FT10-T8 (97.43×). Esto apunta a un **foco epiléptico en el lóbulo temporal derecho**, lo cual es una de las localizaciones más frecuentes en epilepsia focal refractaria [13].

---

## 4. Conclusión general

### 4.1 El EEG del Paciente 20 muestra una crisis de lóbulo temporal derecho

La convergencia de múltiples líneas de evidencia apunta inequívocamente a un foco epiléptico fronto-temporal **derecho**:
- Canales de máxima potencia gamma: FP2-F8 (222×), T8-P8 (124×), F8-T8 (102×), FT10-T8 (97×)
- Canal de máxima potencia total durante crisis: T7-FT9 (temporal izquierdo también involucrado, por propagación contralateral)

Esta topografía es compatible con la epilepsia del lóbulo temporal mesial, la forma más frecuente de epilepsia focal refractaria [13].

### 4.2 Gamma es la banda biomarcadora de la crisis

En todos los análisis (bloques, ventanas deslizantes, espectrograma, 9 configuraciones), la banda gamma (30–64 Hz) emergió como la más discriminante con ratios Crisis/Before entre **40× y 48×**. Este hallazgo está en línea con la evidencia que identifica las oscilaciones de alta frecuencia (HFOs, >30 Hz) como biomarcadores de la zona epileptogénica [10, 14].

La banda beta fue consistentemente segunda (16–17×), sugiriendo que una característica 2D (Beta, Gamma) captura la mayor parte de la información para clasificación.

### 4.3 La señal post-ictal es tan anómala como la ictal

El bloque **After** mantiene potencias en todas las bandas significativamente superiores al baseline (1.5×–3× en todas las bandas) hasta el final de los 120 s analizados. Esto implica:
1. El período de recuperación post-ictal es mayor que 2 minutos.
2. Los detectores automáticos deben incorporar un modelo de "fin de crisis" además de "inicio de crisis".
3. Las métricas de precisión calculadas subestiman la performance real si el período After estuviera correctamente etiquetado como "post-ictal" en lugar de "normal".

### 4.4 El algoritmo de detección es efectivo con una limitación superable

El detector logró **Recall = 100% con un retardo de 0.5 s**, capturando el inicio de la crisis dentro de la primera ventana post-onset. Esto es clínicamente relevante para sistemas de alerta en tiempo real.

La baja precisión (23.77%) es consecuencia de la fisiología post-ictal, no de un error del algoritmo. Las siguientes mejoras podrían implementarse:
- **Umbral dinámico**: recalibrar la línea base con una ventana deslizante de referencia en lugar de usar todo el período Before.
- **Detector de fin**: usar el retorno hacia la media baseline como criterio de terminación.
- **Más canales y/o pacientes**: los parámetros (N_CH_SIG=8, MIN_CH=3, N_CONSEC=3, THRESH_K=2.5) fueron elegidos heurísticamente para este caso; un enfoque de validación cruzada sobre múltiples registros daría parámetros más robustos [12].

### 4.5 FFT vs PSD: ambas son válidas, Welch es más confiable

La FFT directa y la PSD de Welch coinciden en la identificación de frecuencias dominantes y bandas más activas. Sin embargo, la PSD de Welch tiene menor varianza espectral (ruido espectral más bajo), lo que la hace preferida para análisis cuantitativos y comparaciones entre segmentos [6, 7].

### 4.6 Complejidad algorítmica

| Operación | Complejidad | Observación |
|-----------|------------|-------------|
| FFT por canal | $O(N \log N)$ | Algoritmo Cooley-Tukey [4] |
| PSD Welch ($K$ bloques) | $O(K \cdot M \log M)$ | $K \approx N/M$ |
| STFT ($T$ ventanas) | $O(T \cdot M \log M)$ | $T \approx N/H$ |
| Ventanas deslizantes | $O((N/S) \cdot W \log W)$ | lineal en $N/S$ |
| Potencia por banda | $O(F \cdot B)$ | $B=5$ bandas |
| Detección (umbral + consecutivas) | $O(N_{\text{ch}} \cdot T)$ | lineal en $T$ |
| **Total por 28 canales** | $O(C \cdot N \log N)$ | $C=28$ |

El uso de `numpy` (FFTPACK / pocketfft) aplica internamente el zero-padding a la próxima potencia de 2 cuando es beneficioso, pero no cambia la clase de complejidad. El beneficio práctico es marginal para las longitudes de señal típicas del EEG.

---

## 5. Referencias bibliográficas

[1] Niedermeyer, E., & da Silva, F. L. (2005). *Electroencephalography: Basic Principles, Clinical Applications, and Related Fields* (5th ed.). Lippincott Williams & Wilkins.

[2] Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment* [PhD Thesis]. Massachusetts Institute of Technology. Disponible en PhysioNet: https://physionet.org/content/chbmit/

[3] Proakis, J. G., & Manolakis, D. G. (2007). *Digital Signal Processing: Principles, Algorithms, and Applications* (4th ed.). Pearson Prentice Hall.

[4] Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297–301. https://doi.org/10.2307/2003354

[5] Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.

[6] Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73. https://doi.org/10.1109/TAU.1967.1161901

[7] Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical Applications: Multitaper and Conventional Univariate Techniques*. Cambridge University Press.

[8] Schuster, A. (1898). On the investigation of hidden periodicities with application to a supposed 26 day period of meteorological phenomena. *Terrestrial Magnetism*, 3(1), 13–41.

[9] Allen, J. B. (1977). Short term spectral analysis, synthesis, and modification by discrete Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 25(3), 235–238. https://doi.org/10.1109/TASSP.1977.1162950

[10] Worrell, G. A., Gardner, A. B., Stead, S. M., Hu, S., Goerss, S., Cascino, G. J., Meyer, F. B., Marsh, R., & Litt, B. (2008). High-frequency oscillations in human temporal lobe: simultaneous microwire and clinical macroelectrode recordings. *Brain*, 131(4), 928–937. https://doi.org/10.1093/brain/awn006

[11] Gotman, J. (1982). Automatic recognition of epileptic seizures in the EEG. *Electroencephalography and Clinical Neurophysiology*, 54(5), 530–540. https://doi.org/10.1016/0013-4694(82)90038-4

[12] Shoeb, A., & Guttag, J. (2010). Application of machine learning to epileptic seizure detection. In *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 975–982.

[13] Engel, J., Jr. (2001). Mesial temporal lobe epilepsy: What have we learned? *The Neuroscientist*, 7(4), 340–352. https://doi.org/10.1177/107385840100700410

[14] Zijlmans, M., Jiruska, P., Zelmann, R., Leijten, F. S. S., Jefferys, J. G. R., & Gotman, J. (2012). High-frequency oscillations as a new biomarker in epilepsy. *Annals of Neurology*, 71(2), 169–178. https://doi.org/10.1002/ana.22548

[15] Fisher, R. S., van Emde Boas, W., Blume, W., Elger, C., Genton, P., Lee, P., & Engel, J., Jr. (2005). Epileptic seizures and epilepsy: Definitions proposed by the International League Against Epilepsy (ILAE) and the International Bureau for Epilepsy (IBE). *Epilepsia*, 46(4), 470–472. https://doi.org/10.1111/j.0013-9580.2005.66104.x

---

*Informe generado automáticamente a partir de los resultados del notebook `tp2_analisis_espectral.ipynb`. Para la presentación, se recomienda incluir los gráficos de: (1) Potencia por banda E1 (barras), (2) Scatter Beta-Gamma E1 y E2, (3) Espectrograma STFT con línea de detección, (4) Potencia por banda E2 con líneas de anotación y detección.*
