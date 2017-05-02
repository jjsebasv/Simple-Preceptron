#!/usr/bin/python

import random
import numpy as np
from properties import Properties


props = Properties("config.properties")
#Fila corresponde al output, y columna al input
weights = []


def print_weights():
    global weights
    
    for layer in weights:
        print('\n'.join([''.join(['{:4} '.format(item) for item in row]) 
        for row in layer]))
        print(" ")


def init_weights(input_size, output_size):
    global weights
    
    layer_sizes = []
    layer_sizes.append(input_size)
    layer_sizes += props.hidden_layer_sizes
    layer_sizes.append(output_size)

    for i in range(1, len(layer_sizes)):
        prev_layer = layer_sizes[i - 1]
        next_layer = layer_sizes[i]
        weights.append([[random.uniform(-0.5, 0.5) for a in range(prev_layer + 1)] 
            for b in range(next_layer)])

    print(layer_sizes)
    print_weights()
    print("***")


def mapf(arr, f):
    return [f(x) for x in arr]


def forward(input, layer_index):
    global weights
    return np.dot(weights[layer_index], [-1] + input)


def get_outputs(phi, f):
    global weights

    outputs = []
    outputs.append(phi)
    next_input = phi
    for i in range(len(weights)):
        if i >= 1:
            next_input = mapf(next_input, f)
        next_input = forward(next_input, i)
        outputs.append(next_input)
    return outputs


def back_propagation(outputs, expected_output, f, g):
    global props
    global weights

    last_act_output = mapf(outputs[-1], f)
    small_delta = [0 for i in range(len(weights))]
    small_delta[len(weights) - 1] = np.multiply(mapf(outputs[-1], g), np.subtract(expected_output, last_act_output))

    for i in reversed(range(1, len(weights))):
        sum_w_d = np.dot(np.transpose(weights[i][0:-1]), small_delta[i])
        small_delta[i - 1] = np.multiply(mapf(outputs[i], g), sum_w_d)

    for i in range(len(weights)):
        V = outputs[0] if i == 0 else mapf(outputs[i], f)
        V = [-1] + V
        big_delta = np.multiply(props.etha, [np.dot(o_val, V) for o_val in small_delta[i]])
        print(big_delta)
        print(weights[i])
        weights[i] = np.add(weights[i], big_delta)   
    print_weights()


def learn_pattern(phi, expected_output, f, g):
    outputs = get_outputs(phi, f)
    back_propagation(outputs, expected_output, f, g)


def learn_patterns(phis, expected_outputs):
    global weights
    N = 100000
    init_weights(len(phis[0]), len(expected_outputs[0]))

    def f(value):
        return 1 if value >= 0 else -1

    def g(value):
        return value * (1 - value)

    for i in range(N):
        k = random.randint(0, len(phis)-1)
        learn_pattern(phis[k], expected_outputs[k], f, g)

    print(weights)


def main():
    global props

    with open(props.filename) as f:
        lines = f.readlines()

    phis = []
    expected_outputs = []
    for line in lines:
        input = [float(s) for s in line.split("=")[0].split()]
        expected_output = [float(s) for s in line.split("=")[1].split()]

        phis.append(input)
        expected_outputs.append(expected_output)
    #print(phis)
    #print(expected_outputs)   
    learn_patterns(phis, expected_outputs)


if __name__ == "__main__":
    main()
