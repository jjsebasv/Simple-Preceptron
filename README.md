Para correr la red, es necesario tener instalado configparser y numpy.

Para crear los archivos de entrenamiento y testeo a partir de un archivo con datos, se debe setear el archivo de configuración y luego correr: separateInput.py
Para aprender la red, se debe setear el archivo de configuración y correr: main.py

Para interactuar con la red mientras aprende, se pueden apretar las siguientes teclas seguidas de enter:

Q - Da por terminado el aprendizaje.
I - Visualiza el patrón que se esté por correr.
O - Visualiza los outputs para el patrón que se esté corriendo e las distintas capas. El de la capa oculta se encuentra desnormalizado, y por ello se indica "(denorm)".
W - Visualiza los pesos de las distintas capas de la red.
S <filename> - Guarda los pesos de la red en el archivo <filename>.

Ejemplos:

1. Q
	* Termina el aprendizaje de la red.
2. IO
	>: Input: [-2.4522  1.8947], Exp. Output: [-0.7241]
	>: Layer 1 output: [ 0.2283  -0.47306  0.66556]
	>: Layer 2 output (denorm): [ 0.71793]
3. WS pesos_de_red.txt
	>: Weights Layers 0 - 1:
	>: -0.2161 0.1687 0.1108
	>: -6.3118 3.8236 -0.3486
	>: -0.7065 0.5101 0.117
	>:
	>: Weights Layers 1 - 2:
	>: 0.1368 2.0862 0.0817
	* Guarda dichos pesos en el archivo pesos_de_red.txt

El archivo de configuración es config.properties. Los campos son los siguientes:

[Hidden Layers]
sizes: Arreglo con el tamaño de las capas ocultas. Ej: [3, 5]

[Error]
error: Error al que se debe alcanzar con los patrones de entrenamiento para finalizar el aprendizaje.
max_epochs: Máxima cantidad de épocas a correr.
error_freq: Frecuencia/Cantidad de épocas a correr antes de calcular los errores.
file: Archivo donde se guardan los valores de los errores.

[Backpropagation]
#sigmoid or tanh
type: Tipo de la función de backpropagation. Puede ser sigmoid o tanh.
function_beta: Parámetro beta de la función de activación tanto para sigmoid como tanh.
etha: Valor inicial de etha (learning rate).
non_zero_dg: Suma 0.1 a la derivada si se setea en true.
momentum: Utiliza la variación momentum si se setea en true.
adap_etha: Utiliza la variación de etha adaptativo si se setea en true.
momentum_alpha: Alpha de la función de momentum. 
undo_probability: Probabilidad de deshacer el aprendizaje si con el último etha dio un error mayor (Sólo afecta si se usa etha adaptativo).
etha_a: Parámetro "a" de etha adaptativo.
etha_b: Parámetro "b" de etha adaptativo.
epoch_freq: Frecuencia de épocas con las cuales de recalula el etha (Sólo afecta si se usa etha adaptativo).

[Pattern File]
name: Nombre del archivo que contiene todos los patrones.
training: Nombre del archivo que contiene los patrones de entrenamiento.
test: Nombre del archivo que contiene los patrones de testeo.
training_percentage: Porcentaje a tomar de patrones del archivo que contiene a todos para generar el de entrenamiento.
Los restantes se ponen en el archivo de patrones de teseo (test). Utiliza los nombres seteados en training y test para generar dichos archivos. Sólo se usa al correr separateInput.py.

[Weights]
file: Nombre del archivo donde se guardan/leen los pesos.
init_randomly: Inicia los pesos aleatoriamente si se setea en true, y sino se cargan del archivo especificado anteriormente.
save: Guarda los pesos al finalizar el aprendizaje si se setea en true.
sigma: Utiliza una distribución uniforme para los pesos que entran a cada neurona entre -f(m) y f(m), siendo f(m) = m ^ (-1/2) con m el grado de entrada de dicha neurona si se setea esta variable en "middle_nodes". Sino se utiliza una distribución uniforme entre -0.5 y 0.5.

[Function]
file: Archivo donde se guardan los inputs, outputs obtenidos con la red, y outputs esperados para todos los patrones al finalizar el aprendizaje.