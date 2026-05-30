# Practica 2 - Analisis espectral de senales EEG

## 1. Introduccion

El objetivo de esta practica es analizar senales EEG de la base CHB-MIT Scalp EEG Database usando herramientas de analisis espectral. Se busca comparar el comportamiento de la senal antes, durante y despues de una crisis epileptica, identificando cambios en frecuencia, potencia espectral, bandas cerebrales y canales mas relevantes.

El trabajo se organiza en dos escenarios:

- Escenario 1: analisis por bloques separados (`Before`, `Crisis`, `After`).
- Escenario 2: analisis del bloque total concatenado (`Before | Crisis | After`) usando ventanas temporales.

El codigo fue separado en tres archivos:

- `spectral_common.py`: funciones comunes de carga, segmentacion y calculos espectrales.
- `escenario1.py`: analisis por bloques.
- `escenario2.py`: analisis del bloque total.

Las librerias utilizadas son:

```bash
pip install numpy scipy matplotlib pyedflib
```

## 2. Dataset y preparacion de datos

Para el analisis principal se eligio un registro del paciente `chb20`:

```text
chb20_12.edf
```

La informacion de crisis usada fue tomada del archivo `chb20-summary.txt` de CHB-MIT:

| Registro | Inicio crisis | Fin crisis |
|---|---:|---:|
| chb20_12 | 94 s | 123 s |

La frecuencia de muestreo es de 256 Hz y el registro posee 28 canales.

Inicialmente se evaluo la posibilidad de usar tambien otros registros del mismo paciente, pero los graficos resultaban muy similares. Por claridad en la presentacion y para profundizar mejor la interpretacion, se decidio concentrar el informe en `chb20_12`.

La senal fue segmentada en:

```text
Before: 2 minutos antes de la crisis, limitado por el inicio real del archivo.
Crisis: intervalo desde 94 s hasta 123 s.
After: 2 minutos posteriores a la crisis.
Total: concatenacion [Before | Crisis | After].
```

Para `chb20_12`, en la ejecucion se obtuvo:

```text
Before: (28, 24064)
Crisis: (28, 7424)
After: (28, 30720)
Total: (28, 62208)
```

Cada segmento fue centrado por canal restando la media. Esto elimina el offset DC y permite que el analisis espectral se concentre mas en las variaciones de la senal que en desplazamientos constantes.

## 3. Bandas cerebrales consideradas

Para el analisis de potencia por banda se utilizaron los rangos pedidos por el enunciado:

| Banda | Rango |
|---|---:|
| Delta | 0-4 Hz |
| Theta | 4-8 Hz |
| Alpha | 8-12 Hz |
| Beta | 12-30 Hz |
| Gamma | 30-64 Hz |

Estas bandas permiten resumir el contenido espectral de la senal EEG en intervalos fisiologicamente interpretables.

## 4. Escenario 1: analisis por bloques

En el Escenario 1 se analizaron por separado los tres bloques:

```text
Before
Crisis
After
```

La ventaja de este enfoque es que permite comparar de forma directa el comportamiento espectral de cada etapa.

### 4.1 FFT y PSD por bloques

La FFT permite observar la magnitud de las componentes frecuenciales de cada bloque. La PSD por Welch estima la densidad espectral de potencia, promediando ventanas internas y reduciendo la variabilidad respecto de una transformada directa.

En el grafico `Escenario 1 - FFT y PSD por bloques - Todos los canales` se observan dos paneles:

- Arriba: FFT, magnitud espectral.
- Abajo: PSD por Welch, potencia por Hz.

En ambos casos se superponen los 28 canales del EEG como lineas finas y semitransparentes, coloreadas segun el bloque (`Before` en azul, `Crisis` en rojo, `After` en verde). La media entre todos los canales se resalta con una linea gruesa del mismo color. Esta representacion permite observar simultaneamente la variabilidad entre canales y la tendencia promedio de cada etapa.

En la salida por consola se imprimen las frecuencias dominantes. Para el caso analizado:

```text
BEFORE FFT: frecuencias dominantes alrededor de 0.78-1.29 Hz
CRISIS FFT: frecuencias dominantes alrededor de 0.34-0.55 Hz
AFTER FFT: frecuencias dominantes alrededor de 0.17-0.77 Hz
```

En PSD tambien predominan frecuencias bajas:

```text
BEFORE PSD: maximos alrededor de 0.75-1.75 Hz
CRISIS PSD: maximos alrededor de 0.25-1.25 Hz
AFTER PSD: maximos alrededor de 0.25-1.25 Hz
```

Interpretacion:

La mayor energia absoluta aparece en frecuencias bajas. Esto es comun en senales EEG, donde existen componentes lentas, actividad cerebral de baja frecuencia y posibles derivas. Sin embargo, la deteccion de la crisis no debe basarse solamente en la potencia absoluta, sino en cuanto cambia cada banda respecto al estado previo.

En el grafico tambien puede observarse una componente cercana a 60 Hz. Esa frecuencia puede asociarse a interferencia electrica o ruido de linea. En un analisis posterior podria evaluarse un filtro notch en 50/60 Hz, cuidando no alterar informacion util de la banda gamma.

### 4.2 Potencia media por banda cerebral

Luego se calculo la potencia media por banda cerebral usando PSD por Welch y se grafico en barras.

El grafico `Escenario 1 - Potencia media por banda cerebral` compara la potencia promedio de todos los canales en cada banda.

Resultados por consola:

| Banda | Before | Crisis | After | Crisis/Before |
|---|---:|---:|---:|---:|
| Delta | 471.9 | 1608 | 1148 | 3.41x |
| Theta | 85.26 | 220.2 | 131.6 | 2.58x |
| Alpha | 20.12 | 87.23 | 46.36 | 4.34x |
| Beta | 22.54 | 372 | 64.64 | 16.51x |
| Gamma | 6.135 | 298.5 | 52.72 | 48.66x |

Interpretacion:

Delta tiene la potencia absoluta mas alta, especialmente durante la crisis. Esto ocurre porque las bajas frecuencias concentran mucha energia en EEG. Sin embargo, la banda mas discriminante no es necesariamente la de mayor valor absoluto.

Al observar la relacion `Crisis/Before`, las bandas Beta y Gamma aumentan mucho mas que Delta. Gamma aumenta aproximadamente 48.66 veces y Beta 16.51 veces respecto al Before. Esto indica que, para detectar la crisis, Beta y Gamma son mejores indicadores relativos.

Conclusion parcial:

```text
Delta domina en potencia absoluta.
Beta y Gamma discriminan mejor la crisis por aumento relativo.
```

### 4.3 Periodograma por bloques

El periodograma estima la densidad espectral de potencia directamente a partir de la senal completa de cada bloque. A diferencia de Welch, no promedia varias ventanas, por lo que puede mostrar mas variabilidad.

El grafico `Escenario 1 - Periodograma por bloques - Todos los canales` muestra la potencia por frecuencia para `Before`, `Crisis` y `After`. Se superponen los 28 canales con lineas finas y semitransparentes; la media de cada bloque se muestra en negrita y el canal con mayor potencia total se indica con linea discontinua. Esto permite observar el rango de variacion entre canales y cuales se comportan distinto al promedio.

Resultados por consola:

| Bloque | Frec pico prom | Pot total prom | Canal max | Pot canal max |
|---|---:|---:|---|---:|
| Before | 3.09 | 623.7 | P4-O2 | 2202 |
| Crisis | 0.76 | 2685 | T7-FT9 | 8495 |
| After | 0.50 | 1565 | P7-O1 | 5315 |

Interpretacion:

Durante la crisis aumenta claramente la potencia total promedio. El canal con mayor potencia durante la crisis fue `T7-FT9`, lo que sugiere que la actividad epileptica no se expresa de igual manera en todos los canales. Despues de la crisis, la potencia disminuye respecto a la crisis, pero no vuelve inmediatamente al nivel previo, lo que puede representar actividad residual o recuperacion posterior al evento.

### 4.4 STFT por bloques

La STFT permite observar como cambia el contenido frecuencial dentro de cada bloque. En este caso se uso:

```text
Ventana: 2 segundos
Overlap: 50%
```

El grafico `Escenario 1 - STFT por bloques - Promedio de todos los canales` muestra tres espectrogramas, uno para cada bloque. Cada espectrograma es el promedio de potencia entre los 28 canales, lo que suaviza el ruido individual y revela la tendencia espectral general del cerebro en cada etapa. El eje vertical es frecuencia (hasta 64 Hz) y el eje horizontal es tiempo dentro del bloque.

Resultados por consola:

| Bloque | Banda dominante | Canal max | Pot canal max |
|---|---|---|---:|
| Before | Delta | P4-O2 | 671.7 |
| Crisis | Delta | T7-FT9 | 2429 |
| After | Delta | P7-O1 | 1667 |

Potencia media por banda usando STFT:

| Banda | Before | Crisis | After | Crisis/Before |
|---|---:|---:|---:|---:|
| Delta | 167.8 | 625.4 | 444.9 | 3.73x |
| Theta | 30.78 | 82.01 | 48.86 | 2.66x |
| Alpha | 7.272 | 30.48 | 17.63 | 4.19x |
| Beta | 8.444 | 130.2 | 24.5 | 15.43x |
| Gamma | 2.308 | 105.6 | 19.79 | 45.75x |

Interpretacion:

La STFT confirma lo observado con PSD: Delta domina en potencia absoluta, pero Gamma y Beta son las bandas con mayor incremento relativo durante la crisis. La STFT agrega informacion temporal, mostrando que la energia no es constante dentro de los bloques.

### 4.5 Scatter plot por canal

El scatter `Escenario 1 - Scatter por canal: potencia BETA vs GAMMA` representa cada canal como un punto. El eje X corresponde a potencia Beta y el eje Y a potencia Gamma.

Cada bloque tiene un color:

- Before: azul.
- Crisis: rojo.
- After: verde.

Resultados por consola:

| Bloque | Beta prom | Gamma prom | Gamma/Beta | Canal gamma max |
|---|---:|---:|---:|---|
| Before | 22.54 | 6.135 | 0.26 | FT10-T8 |
| Crisis | 372 | 298.5 | 0.55 | FT10-T8 |
| After | 64.64 | 52.72 | 0.55 | FT10-T8 |

Interpretacion:

El scatter muestra que la mayoria de los canales se concentran cerca del origen, mientras que durante la crisis algunos canales se alejan mucho, especialmente `FT10-T8` y `T7-FT9`. Esto indica que la crisis genera un aumento muy marcado de potencia Beta/Gamma en canales especificos.

Es importante notar que en este grafico hay relativamente pocos puntos porque cada punto representa un canal. Con 28 canales y 3 bloques, hay alrededor de 84 puntos. Para obtener mas puntos se usa el Escenario 2, donde cada punto representa una ventana temporal.

## 5. Escenario 2: analisis del bloque total

En el Escenario 2 se trabaja con la senal completa concatenada:

```text
[Before | Crisis | After]
```

Esto permite estudiar la evolucion temporal de la potencia espectral y observar en que momento aparece la crisis dentro del bloque total.

La duracion total analizada fue:

```text
243.0 segundos
```

La crisis queda ubicada dentro del bloque total entre:

```text
94.0 s y 123.0 s
```

### 5.1 FFT y PSD del bloque total

El grafico `Escenario 2 - FFT y PSD del bloque total - Todos los canales` muestra la FFT y la PSD de toda la senal concatenada. Al igual que en el Escenario 1, se representan los 28 canales como lineas finas semitransparentes y la media como linea negra gruesa.

Frecuencias dominantes impresas por consola:

```text
TOTAL FFT:
0.35 Hz, 0.86 Hz, 0.78 Hz, 0.38 Hz, 0.84 Hz

TOTAL PSD:
0.50 Hz, 0.25 Hz, 0.75 Hz, 1.00 Hz, 1.25 Hz
```

Interpretacion:

El bloque total tambien muestra predominio de bajas frecuencias. Esto resume el comportamiento general de toda la senal, pero no permite ubicar temporalmente la crisis. Por eso se complementa con ventanas deslizantes, STFT y espectrogramas.

### 5.2 Potencia por bandas en ventanas de 1 segundo

Se dividio el bloque total en ventanas de:

```text
Ventana: 1 segundo
Paso: 1 segundo
```

Para cada ventana se calculo la potencia media por banda. El grafico `Escenario 2 - Potencia por bandas en ventanas de 1s` muestra cinco curvas, una por banda. La zona de crisis esta marcada en rojo.

Resultados por consola:

| Banda | Before | Crisis | After | Crisis/Before |
|---|---:|---:|---:|---:|
| Delta | 366.3 | 1014 | 758.3 | 2.77x |
| Theta | 80.76 | 221.3 | 112.6 | 2.74x |
| Alpha | 18.28 | 67.37 | 42.31 | 3.69x |
| Beta | 22.05 | 320.2 | 59.51 | 14.53x |
| Gamma | 6.154 | 270.2 | 57.13 | 43.91x |

Interpretacion:

Durante la crisis se observa un aumento claro de potencia en todas las bandas, pero especialmente en Beta y Gamma. Gamma aumenta alrededor de 43.91 veces respecto a Before, y Beta alrededor de 14.53 veces.

Esto confirma que el cambio espectral de la crisis se detecta mejor por aumento relativo en altas frecuencias que por potencia absoluta.

### 5.3 Canales mas relevantes durante la crisis

Durante las ventanas correspondientes a la crisis, se identifico que canales alcanzaron maxima potencia Gamma con mayor frecuencia.

Resultados:

| Canal | Ventanas |
|---|---:|
| FT10-T8 | 11 |
| F8-T8 | 6 |
| FT9-FT10 | 5 |
| T7-FT9 | 5 |
| FP2-F8 | 1 |

Interpretacion:

El canal `FT10-T8` fue el mas frecuente como maximo en Gamma durante la crisis. Esto sugiere que esta derivacion es especialmente informativa para detectar el evento en este archivo. Tambien aparecen canales temporales y frontotemporales, lo que es consistente con una actividad localizada o mas fuerte en ciertas regiones.

### 5.4 Scatter plot por ventanas

El grafico `Escenario 2 - Scatter por ventanas: potencia BETA vs GAMMA` representa cada ventana temporal como un punto. A diferencia del scatter del Escenario 1, aca hay muchos mas puntos porque se grafica una ventana por segundo.

Resultados por consola:

| Estado | Ventanas | Beta prom | Gamma prom |
|---|---:|---:|---:|
| Before | 94 | 22.05 | 6.154 |
| Crisis | 29 | 320.2 | 270.2 |
| After | 120 | 59.51 | 57.13 |

Interpretacion:

Las ventanas de crisis se separan claramente de las ventanas Before. Esto muestra que Beta y Gamma son buenas variables para distinguir el estado de crisis. Las ventanas After quedan en una zona intermedia: disminuyen respecto a la crisis, pero no siempre vuelven inmediatamente a valores bajos.

### 5.5 Periodograma y STFT del bloque total

El grafico `Escenario 2 - Periodograma y STFT del bloque total - Todos los canales` combina:

- Periodograma del bloque total, con los 28 canales superpuestos (lineas finas) y la media en negro.
- STFT del bloque total usando el promedio de todos los canales, con la zona de crisis marcada en rojo.

La STFT permite observar la evolucion temporal del contenido frecuencial. La zona de crisis aparece resaltada, permitiendo comparar visualmente si el aumento de energia coincide con el intervalo anotado. Usar el promedio de canales en la STFT reduce el ruido individual y hace mas visible el patron temporal general.

Resultados de STFT total:

| Banda | Before | Crisis | After | Crisis/Before |
|---|---:|---:|---:|---:|
| Delta | 167.8 | 624.4 | 444.9 | 3.72x |
| Theta | 30.78 | 78.3 | 48.86 | 2.54x |
| Alpha | 7.272 | 29.08 | 17.63 | 4.00x |
| Beta | 8.444 | 123.7 | 24.5 | 14.65x |
| Gamma | 2.308 | 100.8 | 19.79 | 43.66x |

Interpretacion:

La STFT del bloque total confirma el patron de los analisis anteriores: Delta tiene alta potencia absoluta, pero Beta y Gamma muestran el mayor aumento relativo durante la crisis.

### 5.6 Espectrograma por bandas con distintas ventanas y overlaps

El enunciado pide calcular espectrogramas usando al menos 3 ventanas y 3 overlaps. Se probaron:

```text
Ventanas: 1 s, 2 s, 4 s
Overlaps: 25%, 50%, 75%
```

La mejor configuracion se eligio segun el aumento medio `Crisis/Before`.

Resultados principales:

| Ventana | Overlap | Banda | Score | Canal | Score canal |
|---:|---:|---|---:|---|---:|
| 2.0 | 25% | Gamma | 40.87 | FP2-F8 | 242.08 |
| 2.0 | 75% | Gamma | 40.84 | FP2-F8 | 236.46 |
| 4.0 | 25% | Gamma | 40.78 | FP2-F8 | 233.53 |
| 1.0 | 75% | Gamma | 40.62 | FP2-F8 | 230.95 |
| 4.0 | 75% | Gamma | 40.50 | FP2-F8 | 234.76 |
| 1.0 | 25% | Gamma | 40.37 | FP2-F8 | 228.02 |
| 1.0 | 50% | Gamma | 40.30 | FP2-F8 | 231.43 |
| 2.0 | 50% | Gamma | 40.28 | FP2-F8 | 231.57 |
| 4.0 | 50% | Gamma | 39.28 | FP2-F8 | 223.81 |
| 4.0 | 25% | Beta | 17.30 | FT10-T8 | 185.55 |

Interpretacion:

La banda Gamma aparece como la mas discriminante para casi todas las combinaciones de ventana y overlap. La mejor configuracion global fue:

```text
Ventana: 2 segundos
Overlap: 25%
Banda: Gamma
Canal: FP2-F8
```

Esto indica que una ventana intermedia de 2 segundos conserva suficiente resolucion temporal y frecuencial para detectar la crisis. Overlaps mayores tambien funcionan bien, pero aumentan el costo computacional al generar mas ventanas.

## 6. Comparacion general entre escenarios

El Escenario 1 permite una comparacion clara entre bloques. Es util para resumir estadisticamente que ocurre antes, durante y despues de la crisis.

El Escenario 2 permite observar la evolucion temporal. Es mas adecuado para deteccion, porque trabaja con ventanas y permite ver en que momento aparece el aumento de potencia.

En ambos escenarios se repite el mismo resultado principal:

```text
Delta domina en potencia absoluta.
Beta y Gamma son mejores para discriminar la crisis.
```

Esto ocurre porque las bandas altas aumentan proporcionalmente mucho mas durante la crisis que durante el estado previo.

## 7. Frecuencias no deseadas y filtrado

En algunos graficos espectrales puede observarse una componente cercana a 60 Hz. Esta frecuencia podria corresponder a ruido electrico o interferencia de linea. En Argentina la frecuencia de red es 50 Hz, pero dependiendo del origen del dataset, equipo de registro o procesamiento, pueden aparecer componentes en 50/60 Hz.

Una opcion para tratar este problema es aplicar un filtro notch en 50 Hz o 60 Hz. Sin embargo, debe hacerse con cuidado porque la banda Gamma llega hasta 64 Hz. Un filtrado agresivo podria eliminar informacion util de alta frecuencia relacionada con la crisis.

Por este motivo, en este trabajo se identifica la posible frecuencia no deseada, pero no se aplica filtrado automatico para evitar alterar la senal original.

## 8. Complejidad computacional

Sea:

```text
C = cantidad de canales
N = cantidad de muestras
W = cantidad de ventanas
M = muestras por ventana
B = cantidad de bandas
```

### 8.1 Carga y segmentacion

La carga del EDF depende del tamano del archivo. La segmentacion recorre y copia partes de la senal:

```text
Complejidad: O(C * N)
```

### 8.2 Centrado por canal

Para cada canal se calcula la media y se resta:

```text
Complejidad: O(C * N)
```

### 8.3 FFT

La FFT por canal tiene complejidad:

```text
O(N log N)
```

Para todos los canales:

```text
O(C * N log N)
```

### 8.4 PSD por Welch

Welch divide la senal en ventanas y calcula transformadas por ventana. Si hay W ventanas de tamano M:

```text
Complejidad: O(C * W * M log M)
```

### 8.5 Periodograma

El periodograma se basa en una transformada sobre la senal completa:

```text
Complejidad: O(C * N log N)
```

### 8.6 STFT y espectrograma

La STFT y el espectrograma calculan transformadas en ventanas sucesivas:

```text
Complejidad: O(C * W * M log M)
```

Si aumenta el overlap, aumenta W, por lo tanto aumenta el costo computacional.

### 8.7 Potencia por bandas

Para cada banda se integran las frecuencias correspondientes. Si F es la cantidad de frecuencias:

```text
Complejidad: O(C * B * F)
```

En el caso temporal, con ventanas:

```text
Complejidad: O(C * B * F * W)
```

### 8.8 Scatter plots y tablas

Los scatter por canal recorren canales:

```text
Complejidad: O(C)
```

Los scatter por ventana recorren ventanas:

```text
Complejidad: O(W)
```

### 8.9 Uso de random

El uso de una funcion random no mejora la complejidad de este problema. Si se tomaran muestras aleatorias o canales aleatorios, se podria reducir el tiempo de ejecucion, pero se perderia informacion y estabilidad en los resultados. Para detectar una crisis epileptica conviene analizar todos los canales y todas las ventanas disponibles, ya que una seleccion aleatoria podria omitir el canal o intervalo donde la crisis se expresa con mayor claridad.

Conclusion:

```text
Random podria reducir costo en forma aproximada, pero no mejora la complejidad real del algoritmo ni garantiza resultados confiables.
```

## 9. Conclusiones generales

El analisis espectral realizado permite diferenciar claramente la actividad antes, durante y despues de la crisis epileptica.

Los resultados muestran que:

- Durante la crisis aumenta la potencia en todas las bandas.
- Delta presenta la mayor potencia absoluta.
- Beta y Gamma presentan el mayor aumento relativo respecto a Before.
- Gamma es la banda mas discriminante en la mayoria de los analisis.
- Algunos canales, como `FT10-T8`, `T7-FT9`, `F8-T8` y `FP2-F8`, resultan especialmente relevantes.
- El Escenario 1 resume bien las diferencias entre bloques.
- El Escenario 2 permite observar la evolucion temporal y detectar mejor el intervalo de crisis.
- El espectrograma con ventana de 2 segundos y overlap de 25% obtuvo el mejor desempeno en la comparacion realizada.

En consecuencia, para este archivo EEG, la deteccion de la crisis se ve favorecida por el analisis de potencia en bandas Beta y Gamma, especialmente usando ventanas temporales y comparando el aumento relativo respecto al segmento previo a la crisis.

## 10. Estructura y ejecucion del proyecto

La estructura recomendada del repositorio es:

```text
repo/
  archivos/
    chb20_12.edf
  tp2_analisis_espectral/
    escenario1.py
    escenario2.py
    spectral_common.py
    informe.md
```

Los archivos EDF no necesariamente deben subirse a GitHub si son pesados, pero deben estar localmente en:

```text
repo/archivos/
```

Para ejecutar el Escenario 1:

```bash
cd tp2_analisis_espectral
py -3.12 escenario1.py
```

Para ejecutar el Escenario 2:

```bash
py -3.12 escenario2.py
```

El codigo busca automaticamente cada EDF primero en:

```text
../archivos/
```

Y como segunda opcion en:

```text
./archivos/
```

Si se utiliza otro archivo o una crisis distinta, se debe modificar el diccionario `RECORDS` en `spectral_common.py`:

```python
RECORDS = {
    "chb20_12": {"file": "chb20_12.edf", "start_sec": 94, "end_sec": 123},
}
```
