# TODO
## En orden de prioridad

* ~Elegir subconjunto de patrones p/ entrenamiento y el resto para testeo~
  * ~porcentual `config.properties`~
  * ~dividir el archivo de entrada en dos --> Entrenamiento y testeo.~

* ~Normalizar entrada~
* ~Normalizar salida~

* ~Tomar todos los patrones en una época~
  * ~Ordenarlos aleatoriamente y tomarlos "en orden"~

* ~Sumar los `delta_weight` en batch~
  * ~Guardar los `delta_weight` por input y sumarlos entre si.~
  * ~Actualizar los pesos de los nodos, para la segunda corrida (con el segundo epoca)~

* ~Graficar error de testeo y de entrenamiento (sobre las corridas de epocas)~
  * ~`Sum(1/2N (si - oi)^2)`~
  * ~Escupir los errores a un archivo~

* ~3 variantes de backpropagation~
  * ~etha adaptativo~
  * ~0.1~
  * ~Momentum~

* ~Visualiar salida en capas ocultas~
* ~Poder elegir la funcion de activacion (`config.properties`)~
* ~Poder agregar una última capa lineal~

* Inicializar pesos adecuadamente
  * Actualizar el README.md
  *  ~reconmendado: `sigma(m) = m ^ (-1/2)`~

* ~Poder configuar la frecuencia con la que se calcula el error (cada cuantas epocas)~

* ~Guardar los pesos finales en un archivo~
  * ~Levantarlos de nuevo --> en vez de pesos iniciales random~

* Comparar tanh con exp

* Comparar distintas arquitecturas. Yo (Tom) propongo:
  * Sin capas ocultas
  * Capa oculta de 10 neuronas
  * Capa oculta de 30 neuronas
  * 3 Capas ocultas de 10 neuronas cada una
  * 3 Capas ocultas de 20 neuronas cada una
  * 5 Capas ocultas de 10 neuronas cada una
  * 5 Capas ocultad de 20 neuronas cada una

* Graficar la función del terreno basado en inputs vs generada por la red

* Graficar error de testeo y entrenamiento en mismo grafico

* Comparar sin variaciones, con eta adaptativo, con momentum y con eta + momentum. (Yo dejaría la de la derivada siempre on)

* Correr para distintas arquitecturas, y distintas variaciones

