#!/usr/bin/python3

import collections
import numpy as np
import random
from threading import Thread
from functools import reduce
from properties import Properties


props = Properties("config.properties")


class NeuralNetwork:

    def __init__(self, train_patterns, test_patterns, etha):
        # Row corresponds to output, and column to input
        self.layers_weights = []
        self.saved_weights = None
        self.etha = etha
        [self.input_patterns, self.test_patterns] = self.normalize_patterns(train_patterns, test_patterns)
        self.delta_weights = []
        self.prev_delta_weights = []
        self.prev_prev_sqr_error = 0
        self.prev_sqr_error = 0
        self.sqr_error = 0
        self.training_errors = []
        self.test_errors = []

        self.stop = False
        self.view_weights = False
        self.view_pattern = False
        self.view_outputs = False
        self.save_weights = False
        self.weights_file = ""

    def tanh(self, x):
        return np.tanh(x * props.beta)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-2 * props.beta * x))

    def d_tanh(self, x):
        return props.beta * (1 - x * x)

    def d_sigmoid(self, x):
        return 2 * props.beta * x * (1 - x)

    def normalize_patterns(self, train_patterns, test_patterns):
        outputs = reduce(lambda x, y: x + y, [pattern.expected_output for pattern in train_patterns], [])
        inputs = reduce(lambda x, y: x + y, [pattern.input for pattern in train_patterns], [])
        
        def norm(v, a, b, c, d):
            return (d - c) * v / (b - a) + d - ((d - c) * b / (b - a)) 
        
        o_min = 0.1 if props.function_type == "sigmoid" else -0.8
        o_max = 0.9 if props.function_type == "sigmoid" else 0.8

        self.norm_input = lambda x: norm(x, min(inputs), max(inputs), -1 , 1)
        self.denorm_input = lambda x: norm(x, -1 , 1, min(inputs), max(inputs))
                    
        def norm_output(v):
            return norm(v, min(outputs), max(outputs), o_min, o_max)

        self.denorm_output = lambda x: norm(x, o_min, o_max, min(outputs), max(outputs))

        norm_train_patterns = []
        for pattern in train_patterns:
            new_input = [self.norm_input(v) for v in pattern.input]
            new_output = [norm_output(v) for v in pattern.expected_output]
            norm_train_patterns.append(Pattern(new_input, new_output))

        norm_test_patterns = []
        for pattern in test_patterns:
            new_input = [self.norm_input(v) for v in pattern.input]
            new_output = [norm_output(v) for v in pattern.expected_output]
            norm_test_patterns.append(Pattern(new_input, new_output))

        return [norm_train_patterns, norm_test_patterns]

    def init_weights(self, layer_sizes):
        layer_sizes_len = len(layer_sizes)
        
        if not props.init_w_randomly:
            self.layers_weights = self.read_weights()
            if layer_sizes_len - 1 != len(self.layers_weights):
                raise Exception("{} layers doesn't match {}".format(len(self.layers_weights), layer_sizes_len))
            for i in range(layer_sizes_len - 1):
                layer_shape = (layer_sizes[i + 1], layer_sizes[i] + 1)
                if layer_shape != np.shape(self.layers_weights[i]):
                    raise Exception("{} layer size doesn't match {}".format(np.shape(self.layers_weights[i]), layer_shape))
            return

        def random_uniform_list(n):
            return [random.uniform(-0.5, 0.5) for _ in range(n)]

        for i in range(1, layer_sizes_len):
            prev_layer_size = layer_sizes[i - 1] + 1 # Considering bias node too
            curr_layer_size = layer_sizes[i]
            weights = [random_uniform_list(prev_layer_size) for b in range(curr_layer_size)]
            self.layers_weights.append(weights)

    def read_weights(self):
        layer_weights = []
        with open(props.weights_file) as f:
            i = 0
            lines = f.readlines()
            layers = int(lines[i])
            i += 1
            for j in range(layers):
                rows = int(lines[i].split()[0])
                i += 1
                layer = [[] for r in range(rows)]
                for r in range(rows):
                    layer[r] = [float(x) for x in lines[i].split()]
                    i += 1
                layer_weights.append(layer)
        return layer_weights

    def write_weights(self, layer_weights):
        with open(props.weights_file, "w+") as f:
            f.write("{}\n".format(len(layer_weights)))
            for layer_w in layer_weights:
                f.write("{} {}\n".format(len(layer_w), len(layer_w[0])))
                for row in layer_w:
                    f.write(" ".join(str(x) for x in row.tolist()))
                    f.write("\n")

    def get_g(self):
        return self.sigmoid if props.function_type == "sigmoid" else self.tanh

    def get_dg(self):
        return self.d_sigmoid if props.function_type == "sigmoid" else self.d_tanh

    def learn_patterns(self, n):
        g = self.get_g()
        dg = self.get_dg()

        self.prev_delta_weights = [np.zeros(np.shape(layer)) for layer in self.layers_weights]
        delta_error = 0
        self.finish = False
        thread = Thread(target = self.read_stdin, args = [self])
        thread.start()

        for epoch in range(n):
            self.check_view_weights()
            self.check_save_weights()
            self.calculate_error(epoch)
            # If error reached or Q was pressed, break
            if (self.sqr_error < props.error and epoch > 100) or self.stop:
                break
            self.reset_error_counters()
            self.run_epoch(g, dg)         
            delta_error = self.sqr_error - self.prev_sqr_error            
            shouldUndo = self.adaptative_etha(delta_error, epoch)
            if shouldUndo:
                continue
            for i, _ in enumerate(self.layers_weights):
                self.layers_weights[i] = np.add(self.layers_weights[i], self.delta_weights[i])
                self.momentum(delta_error, i)
            self.prev_delta_weights = self.delta_weights
        self.finish = True

        if props.save_weights:
            self.write_weights(self.layers_weights)

        self.write_error()

    def read_stdin(self, args):
        network = args
        print("Press Q and Enter to Quit.\n"
                 + "Press W to view weights.\n"
                 + "Press I to view pattern.\n"
                 + "Press O to view outputs.\n"
                 + "Press 'S <filename>' to save the weights.\n")
        while not network.stop and not network.finish:
            while network.save_weights:
                None    
            key = input()
            if key == "Q":
                network.stop = True
            if "W" in key:
                network.view_weights = True
            if "I" in key:
                network.view_pattern = True
            if "O" in key:
                network.view_outputs = True
            if "S" in key:
                network.save_weights = True
                network.weights_file = key.split("S ")[1]
            key = ""

    def check_save_weights(self):
        if self.save_weights:
            aux = props.weights_file
            props.weights_file = self.weights_file
            self.write_weights(self.layers_weights)
            props.weights_file = aux
            self.save_weights = False

    def check_view_weights(self):
        if self.view_weights:
            for i in range(len(self.layers_weights)):
                print("\nWeights Layers {} - {}:".format(i, i + 1))
                self.print_weights(i)
            self.view_weights = False

    def print_weights(self, i):
        for row in self.layers_weights[i]:
            print(" ".join(str(x) for x in row.tolist()))
            print("\n")

    def calculate_error(self, epoch):
        if epoch % props.error_freq == 0:
            self.training_errors.append(self.sqr_error)
            self.test_errors.append(self.get_test_error())
            print("Epoch: {}, Training: {}, Test: {}".format(epoch, self.training_errors[-1], self.test_errors[-1]))

    def write_error(self):
        if props.error_file != None and props.error_file != "":
            with open(props.error_file, "w+") as err_f:
                for i in range(len(self.training_errors)):
                    training = self.training_errors[i]
                    test = self.test_errors[i]
                    err_f.write("{};{};{}\n".format(i * props.error_freq, training, test))

    def reset_error_counters(self):
        self.prev_prev_sqr_error = self.prev_sqr_error
        self.prev_sqr_error = self.sqr_error
        self.sqr_error = 0
        self.delta_weights = [np.zeros(np.shape(layer)) for layer in self.layers_weights]

    def adaptative_etha(self, delta_error, epoch):
        if props.use_adap_etha and epoch % props.epoch_freq == 0:
            self.etha += self.get_delta_etha(delta_error)
            if delta_error > 0 and random.random() <= props.undo_probability and self.saved_weights != None:
                self.layers_weights = self.saved_weights
                self.prev_delta_weights = [np.zeros(np.shape(layer)) for layer in self.layers_weights]
                self.prev_sqr_error = self.prev_prev_sqr_error
                self.sqr_error = self.prev_sqr_error
                return True
            else:
                self.saved_weights = self.layers_weights
        return False

    def get_delta_etha(self, delta_error):
        if delta_error <= 0:
            return props.etha_a
        else:
            return -props.etha_b * self.etha

    def momentum(self, delta_error, i):
        if props.use_momentum and delta_error <= 0:
            delta_momentum = np.multiply(props.momentum_alpha, self.prev_delta_weights[i])
            self.layers_weights[i] = np.add(self.layers_weights[i], delta_momentum)

    def run_epoch(self, g, dg):
        random.shuffle(self.input_patterns)
        for pattern in self.input_patterns:
            self.check_view_pattern(pattern)
            self.learn_pattern(pattern, g, dg)
        self.sqr_error = self.sqr_error / (2 * len(self.input_patterns))

    def check_view_pattern(self, pattern):
        if self.view_pattern:
            i = [self.denorm_input(o_val) for o_val in pattern.input]
            o = [self.denorm_output(o_val) for o_val in pattern.expected_output]
            print("\nInput: {}, Exp. Output: {}".format(np.round(i, 5), np.round(o, 5)))
            self.view_pattern = False

    def learn_pattern(self, pattern, g, dg):
        outputs = self.get_outputs(pattern.input, g)
        self.check_view_outputs(outputs)
        self.backpropagate(outputs, pattern.expected_output, g, dg)

    def check_view_outputs(self, outputs):
        if self.view_outputs:
            for i in range(len(outputs)):
                if i == len(outputs) - 1:
                    output = o = [self.denorm_output(o_val) for o_val in outputs[i]]
                    print("\nLayer {} output (denorm): {}".format(i, np.round(outputs[i], 5)))
                elif i >= 1:
                    print("\nLayer {} output: {}".format(i, np.round(outputs[i], 5)))
            self.view_outputs = False

    def get_outputs(self, input, g):
        outputs = [input]
        next_input = input

        for i, weight in enumerate(self.layers_weights):
            forwarded_values = self.forward(next_input, weight)
            next_input = [g(x) for x in forwarded_values]
            outputs.append(next_input)

        return outputs

    def get_test_error(self):
        if len(self.test_patterns) == 0:
            return None
        error = 0
        for pattern in self.test_patterns:
            output = self.get_outputs(pattern.input, self.get_g())[-1]
            error = sum(np.power(np.subtract(pattern.expected_output, output), 2))
        return error / (2 * len(self.test_patterns))

    def get_output(self, pattern):
        norm_pattern = [self.norm_input(i_val) for i_val in pattern]
        output = self.get_outputs(norm_pattern, self.get_g())[-1]
        return [self.denorm_output(o_val) for o_val in output]

    def forward(self, input, layer_weights):
        return np.dot(layer_weights, input + [-1])

    def backpropagate(self, outputs, expected_output, g, dg):
        layers_weights_len = len(self.layers_weights)
        small_delta = [0 for _ in self.layers_weights]
        negl = 0.1 if props.use_non_zero_dg else 0
        dgs = [dg(x) + negl for x in outputs[-1]]
        expected_difference = np.subtract(expected_output, outputs[-1])
        self.sqr_error += sum(np.power(expected_difference, 2))
        small_delta[layers_weights_len - 1] = np.multiply(dgs, expected_difference)

        for i in reversed(range(1, layers_weights_len)):
            sum_w_d = np.dot(np.transpose(self.layers_weights[i])[0:-1], small_delta[i])
            current_dgs = [dg(x) + negl for x in outputs[i]]
            small_delta[i - 1] = np.multiply(current_dgs, sum_w_d)

        for i, _ in enumerate(self.layers_weights):
            V = outputs[0] if i == 0 else [g(x) for x in outputs[i]]
            V = np.array(V + [-1])
            #Convert arrays into vector-like matrices
            V = np.array(V).reshape(len(V), 1)
            small_delta[i] = np.array(small_delta[i]).reshape(len(small_delta[i]), 1)
            layer_delta_weights = np.multiply(self.etha, np.dot(small_delta[i], np.transpose(V)))
            self.delta_weights[i] = np.add(self.delta_weights[i], layer_delta_weights)

Pattern = collections.namedtuple('Pattern', ['input', 'expected_output'])

def read_patterns(f):
    lines = f.readlines()[1:]

    patterns = []
    for line in lines:
        aux = line.split()
        inputs = aux[0:-1]
        expected_outputs = [aux[-1]]
        input_values = [float(x) for x in inputs]
        expected_outputs_values = [float(x) for x in expected_outputs]
        patterns.append(Pattern(input_values, expected_outputs_values))
    return patterns

def main():
    props = Properties("config.properties")

    train_patterns = []
    with open(props.training_file) as f:
        train_patterns = read_patterns(f)

    input_size = len(train_patterns[0].input)
    output_size = len(train_patterns[0].expected_output)
    layers_sizes = [input_size] + props.hidden_layer_sizes + [output_size]

    test_patterns = []
    if props.test_file != "":
        with open(props.test_file) as f:
            test_patterns = read_patterns(f)

    network = NeuralNetwork(train_patterns, test_patterns, props.etha)
    network.init_weights(layers_sizes)   
    network.learn_patterns(props.max_epochs)

    all_patterns = []
    with open(props.filename) as f:
        all_patterns = read_patterns(f)

    for pattern in all_patterns:
        print("{} | {}".format(network.get_output(pattern.input), pattern.expected_output))


    # # Checking that everything works as intended
    # if input_size == 2:
    #     print(network.get_output([1, 1]))
    #     print(network.get_output([1, -1]))
    #     print(network.get_output([-1, 1]))
    #     print(network.get_output([-1, -1]))
    # else:
    #     print(network.get_output([0])) #~0
    #     print(network.get_output([0.1])) #~1
    #     print(network.get_output([0.3])) #2
    #     print(network.get_output([0.4])) #5
    #     print(network.get_output([0.45])) #2
    #     print(network.get_output([0.475])) #~1
    #     print(network.get_output([0.5])) #0
    #     print(network.get_output([0.55])) #3
    #     print(network.get_output([0.6])) #~20
    #     print(network.get_output([0.65])) #~28
    #     print(network.get_output([0.7])) #33
    #     print(network.get_output([0.75])) #~40
    #     print(network.get_output([0.85])) #~45
    #     print(network.get_output([0.8])) #50
    #     print(network.get_output([0.9])) #78
    #     print(network.get_output([1])) #100


if __name__ == "__main__":
    main()
