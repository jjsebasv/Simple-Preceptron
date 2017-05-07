#!/usr/bin/python

import collections
import numpy as np
import random
from properties import Properties


props = Properties("config.properties")
#Fila corresponde al output, y columna al input
weights = []


class NeuralNetwork:

    def __init__(self, patterns, etha):
        self.layers_weights = []
        self.etha = etha
        self.input_patterns = patterns

    def init_weights(self, layer_sizes):
        layer_sizes_len = len(layer_sizes)

        def random_uniform_list(n):
            return [random.uniform(-0.5, 0.5) for _ in range(n)]

        for i in range(1, layer_sizes_len):
            prev_layer_size = layer_sizes[i - 1] + 1 # Considering bias node too
            curr_layer_size = layer_sizes[i]
            weights = [random_uniform_list(prev_layer_size) for b in range(curr_layer_size)]
            self.layers_weights.append(weights)

    def learn_patterns(self, n):
        def g(x):
            return np.tanh(x)

        def dg(x):
            return 1 - x * x

        for _ in range(n):
            pattern = random.choice(self.input_patterns)
            self.learn_pattern(pattern, g, dg)

    def learn_pattern(self, pattern, g, dg):
        outputs = self.get_outputs(pattern.input, g)
        self.backpropagate(outputs, pattern.expected_output, g, dg)

    def get_outputs(self, input, g):
        outputs = [input]
        next_input = input

        for i, weight in enumerate(self.layers_weights):
            forwarded_values = self.forward(next_input, weight)
            next_input = [g(x) for x in forwarded_values]
            outputs.append(next_input)

        return outputs

    def forward(self, input, layer_weights):
        return np.dot(layer_weights, input + [-1])

    def backpropagate(self, outputs, expected_output, g, dg):
        layers_weights_len = len(self.layers_weights)
        small_delta = [0 for _ in self.layers_weights]
        dgs = [dg(x) for x in outputs[-1]]
        expected_difference = np.subtract(expected_output, outputs[-1])
        small_delta[layers_weights_len - 1] = np.multiply(dgs, expected_difference)

        for i in reversed(range(1, layers_weights_len)):
            sum_w_d = np.dot(np.transpose(self.layers_weights[i])[0:-1], small_delta[i])
            current_dgs = [dg(x) for x in outputs[i]]
            small_delta[i - 1] = np.multiply(current_dgs, sum_w_d)

        for i, _ in enumerate(self.layers_weights):
            V = outputs[0] if i == 0 else [g(x) for x in outputs[i]]
            V = V + [-1]
            big_delta = np.multiply(self.etha, [np.dot(o_val, V) for o_val in small_delta[i]])
            self.layers_weights[i] = np.add(self.layers_weights[i], big_delta)

Pattern = collections.namedtuple('Pattern', ['input', 'expected_output'])

def main():
    props = Properties("config.properties")

    with open(props.filename) as f:
        lines = f.readlines()[1:]

    patterns = []
    for line in lines:
        aux = line.split()
        inputs = [aux[0], aux[1]]
        expected_outputs = [aux[2]]
        input_values = [float(x) for x in inputs]
        expected_outputs_values = [float(x) for x in expected_outputs]
        patterns.append(Pattern(input_values, expected_outputs_values))

    input_size = len(patterns[0].input)
    output_size = len(patterns[0].expected_output)
    layers_sizes = [input_size] + props.hidden_layer_sizes + [output_size]

    network = NeuralNetwork(patterns, props.etha)
    network.init_weights(layers_sizes)
    network.learn_patterns(10000)

    # Checking that everything works as intended

    def f(x):
        return np.tanh(x)

    print(network.get_outputs([1, 1], f)[-1])
    print(network.get_outputs([1, -1], f)[-1])
    print(network.get_outputs([-1, 1], f)[-1])
    print(network.get_outputs([-1, -1], f)[-1])

#
# def learn_patterns(phis, expected_outputs):
#     global weights
#     N = 3000
#     init_weights(len(phis[0]), len(expected_outputs[0]))
#
#     # def f(x):
#     #     return 1 if x >= 0 else -1
#     #
#     # def g(x):
#     #     return 1
#     #
#     # def f(x):
#     #     return 1 / (1 + np.exp(-x))
#
#     # def g(x):
#     #     return x * (1 - x)
#
#     def f(x):
#         return np.tanh(x)
#
#     def g(x):
#         return 1 - x * x
#
#     for i in range(N):
#         k = random.randint(0, len(phis)-1)
#         learn_pattern(phis[k], expected_outputs[k], f, g)
#
#     print(weights)
#
#
# def main():
#     global props
#
#     with open(props.filename) as f:
#         lines = f.readlines()
#
#     def f(x):
#         return np.tanh(x)
#
#     phis = []
#     expected_outputs = []
#     for line in lines:
#         input = [float(s) for s in line.split("=")[0].split()]
#         expected_output = [float(s) for s in line.split("=")[1].split()]
#
#         phis.append(input)
#         expected_outputs.append(expected_output)
#     #print(phis)
#     #print(expected_outputs)
#     learn_patterns(phis, expected_outputs)
#
#     print(get_outputs([1, 1], f)[-1])
#     print(get_outputs([1, -1], f)[-1])
#     print(get_outputs([-1, 1], f)[-1])
#     print(get_outputs([-1, -1], f)[-1])


if __name__ == "__main__":
    main()
