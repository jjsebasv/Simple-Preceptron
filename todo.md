# TODO
## En orden de prioridad

* Elegir subconjunto de patrones p/ entrenamiento y el resto para testeo
  * ~~ porcentual `config.properties`
  * dividir el archivo de entrada en dos --> Entrenamiento y testeo.

* Normalizar entrada
* Normalizar salida

* Tomar todos los patrones en una época
  * Ordenarlos aleatoriamente y tomarlos "en orden"

* Sumar los `delta_weight` en batch
  * Guardar los `delta_weight` por input y sumarlos entre si.
  * Actualizar los pesos de los nodos, para la segunda corrida (con el segundo epoca)

* Graficar error de testeo y de entrenamiento (sobre las corridas de epocas)
  * `Sum(1/2N (si - oi)^2)`

* 3 variantes de backpropagation
* etha adaptativo

* Visualiar salida en capas ocultas
* Poder elegir la funcion de activacion (`config.properties`)
* Poder agregar una última capa lineal

* Inicializar pesos adecuadamente
  *  reconmendado: `sigma(m) = m ^ (-1/2)`

* Poder configuar la frecuencia con la que se calcula el error (cada cuantas epocas)

* Guardar los pesos finales en un archivo
  * Levantarlos de nuevo --> en vez de pesos iniciales random
