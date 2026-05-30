

## Algor
## ́
ıtmica y l
## ́
ogica computacional
Departamento de inform
## ́
atica UCA
## Pr
## ́
actica 2 - An
## ́
alisis espectral
## 1
La finalidad de esta pr
## ́
actica es trabajar con series temporales haciendo an
## ́
alisis
espectrales de las se
## ̃
nales para encontrar informaci
## ́
on valiosa de las se
## ̃
nales.
Se trabajar
## ́
a con la base de datos CHBMIT Scalp EEG Database de physionet Lea
bien las instrucciones de las anotaciones de los archivos.
-  Cada archivo tiene la extension .edf∴es necesario convertirlo a una lista o matriz
para poder analizarlo→ver como funciona la librer
## ́
ıapyedflib, junto conimport
highlevel.
-  Para cada archivo se tiene una crisis epil
## ́
eptica anotada i.e.inicio y finde la crisis,
e.g. el archivoc ̧hb01-summary.txt”tiene toda la informaci
## ́
oon necesaria para la
carpetachb01, e.g. frecuencia de muestreo, los canales usados y las crisis: File
Name: chb01 03.edf
## File Start Time: 13:43:04
## File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
Note que todo est
## ́
a en segundos, as
## ́
ı que se tiene que trabajar correctamente de
acuerdo a la frecuencia de muestreo.
Para convertir de muestras a segundos, basta con hacer una regla de 3
Si  1  Hz  equivale  a  1  segundo,  cu
## ́
antos  segundos  equivalen  a  la  frecuencia  de
muestreo en Hz.
Moverse a trav
## ́
es de un cicloforen pasos de 1, corresponde a moverse en pasos
de una muestra o en pasos de la frecuencia de muestreo.
-  Segmente  las  se
## ̃
nales  de  tal  manera  que  el  total  de  la  se
## ̃
nal  a  analizar  est
## ́
e
compuesta por 2 minutos antes de la crisis, crisis y 2 minutos despu
## ́
es de la crisis,
e.g. si la crisis dura un minuto, la duraci
## ́
oon de la se
## ̃
nal para trabajar ser
## ́
a de 5
minutos: 2 min antes + 1 min crisis + 2 min despu
## ́
es. Nota: Restarle la media a
cada segmento por canal, hace que se centre todas las se
## ̃
nales.
-  Se analizar
## ́
an dos escenarios
Escenario 1 Por Bloques: Para analizar cada bloque del paso anterior, i.e.
Before = segmento de 2m de duraci
## ́
on antes de crisis.
Crisis = segmento de 1m de duraci
## ́
on de la crisis.
After = segmento de 2 m de duraci
## ́
on despu
## ́
es de la crisis.

Escenario 2 Bloque total: para analizar la se
## ̃
nal moviendose en pasos de frecuen-
cia de muestreo o en segundos, i.e.
Se tiene la se
## ̃
nal completa compuesta por [Before Crisis After]. Observe que los
bloques est
## ́
an seguidos, no est
## ́
an divididos.
Para cada escenario se pide:
-  Estime la FFT y la PSD. Compare ambos m
## ́
etodos. Determine visualmente si se
observan frecuencias no deseadas. En caso de que existan, investigue si es posible
filtrarlas de tal manera que no se altere la se
## ̃
nal.
-  Calcule el espectrograma por rango de frecuencias cerebrales en Hz, i.e.
Fdelta = 0-4Hz
Ftheta = 4-8Hz
Falpha = 8-12Hz
Fbeta = 12-30Hz
Fgamma = 30-64Hz;
...usando al menos 3 ventanas y 3 diferentes overlaping.
Determine cu
## ́
ales son las mejores opciones para detectar la crisis y para cu
## ́
ales
canales?
Grafique por medio de unscaterploty determine diferencias entre cada segmento.
- Estime el periodograma y la STFT. Gr
## ́
afique y saque conclusiones.
- Para cada rutina, determine su complejidad. Analice si usar una funci
## ́
onram-
dom, mejora la complejidad.
Tener en cuenta:
Esta pr
## ́
actica se defender
## ́
a en clase por todo el grupo. Cualquier persona puede
hacer preguntas de la exposici
## ́
on.
Debe entregar los c
## ́
odigos junto con todas las l
## ́
ıibrerias que se instalaron comen-
tadas, e.g.
pip install pyedflib from pyedflib import highlevel
Cabe resaltar que todo el c
## ́
odigo debe estar debidamente explicado.
Debe entregar un informe, completo y detallado, de todo lo desarrollado, expli-
cando muy bien y claro, todos los resultados obtenidos.
Si desea trabajar en latex, puede hacer un usuario en overleaf y copiar el siguiente
c
## ́
odigo overleaf
Una presentaci
## ́
on con la informaci
## ́
on m
## ́
as relevante, siempre ayuda a una mejor
comprensi
## ́
on.
Consultas: no duden en escribirme al correo uca.
## 2

Ejemplo de como cargar las se
## ̃
nales
En este c
## ́
oodigo de MATLAB, se toma el mismo tiempo de la crisis, antes y despu
## ́
es, e.g.
si la crisis dura 20 seg, entonces se tomaran 20 seg antes de la crisis y 20 seg despu
## ́
es de
la crisis. Tambi
## ́
en se puede hacer en json.
A  ={( ’ c h b 0 1
## 1 8  ’ ) ,
( ’ c h b 0 12 6  ’ ) ,
( ’ c h b 0 3
## 3 5  ’ ) ,
( ’ c h b 0 3
## 3 6  ’ ) ,
( ’ c h b 0 51 7  ’ ) ,
( ’ c h b 0 52 2  ’ ) ,
( ’ c h b 0 71 2  ’ ) ,
( ’ c h b 0 71 9  ’ ) ,
( ’ c h b 0 90 6  ’ ) ,
( ’ c h b 1 02 7  ’ ) ,
( ’ c h b 1 03 1  ’ ) ,
( ’ c h b 1 1
## 9 2  ’ )};
%V e c t o r   c o n   t o d o s    l o s    S e i z u r e    S t a r t .
s t a r t S e i z u r e   =   [ 4 4 0 3 2 0 , 4 7 6 6 7 2 , 6 6 3 5 5 2 , 4 4 1 6 0 0 , 6 2 7 4 5 6 , 6 0 1 0 8 8 , 1 2 5 9 5 2 0 , . . .
. . . 3 5 0 4 1 2 8 , 3 1 3 1 1 3 6 , 6 0 9 7 9 2 , 9 7 3 0 5 6 , 6 8 9 9 2 0 ] ;
%V e c t o r   c o n   t o d o s    l o s    S e i z u r e   End .
e n d S e i z u r e   =   [ 4 6 3 3 6 0 , 5 0 2 5 2 8 , 6 7 9 9 3 6 , 4 5 5 1 6 8 , 6 5 8 1 7 6 , 6 3 1 0 4 0 , 1 2 8 1 5 3 6 , . . .
. . . 3 5 4 0 7 3 6 , 3 1 4 7 5 2 0 , 6 2 6 4 3 2 , 9 9 2 5 1 2 , 6 9 8 1 1 2 ] ;
%D u r a c i o n
d u r a t i o n   =   e n d S e i z u r e − s t a r t S e i z u r e ;
%C a r g a r    l a    s e n i a l
f o r   B  =   1 : 1 2
d a t a   =   l o a d (A{B}) ;
v a r i a b l e s = f i e l d s ( d a t a ) ;
X  =   d a t a . (  v a r i a b l e s{1}) ;
%L o n g i t u d   d e   l a    s e n i a l
l e n g   =   l e n g t h (X ) ;
%B u s c o   d i m e n s i o n   d e   c a n a l e s    i n d e p e n d i e n t e s .
dim   =   d i m e n s i o n (X ) ;
%S e p a r o   e n   S e i z u r e  ,   non − s e i z u r e .
N o n S e i z u r e B e f o r e   =  X ( :  ,  s t a r t S e i z u r e ( B) − d u r a t i o n ( B ) :  s t a r t S e i z u r e ( B ) ) ;
S e i z u r e=  X ( :  ,  s t a r t S e i z u r e ( B ) : e n d S e i z u r e ( B ) ) ;
N o n S e i z u r e A f t e r   =  X ( :  , e n d S e i z u r e ( B ) : e n d S e i z u r e ( B) + d u r a t i o n ( B ) ) ;
e n d
## 3