import random
import numpy


def get_weights(n):
    weights = []
    for i in range(n):
        weights.append(random.uniform(-0.5, 0.5))
    return weights


def dot(v1, v2):
    acum = 0
    for i in range(len(v1)):
        acum += v1[i] * v2[i]
    return acum


def get_output(phi, weights, f):
    return f(dot(phi, weights))


def learn_pattern(phi, expected_output, weights, f, etha):
    output = get_output(phi, weights, f)
    correction = [val * etha * (expected_output - output) for val in phi]
    return [weights[i] + correction[i] for i in range(len(weights))]


def learn_patterns(phis, expected_outputs):
    N = 100000
    etha = 0.05
    weights = get_weights(len(phis[0]))
    def f(value):
        if value >= 0:
            return 1
        else:
            return -1
    for i in range(N):
        k = random.randint(0, len(phis)-1)
        weights = learn_pattern(phis[k], expected_outputs[k], weights, f, etha)
    print(weights)
    print(dot(weights, [0.5, -1]))
    print(dot(weights, [0.65, -1]))
    print(dot(weights, [0.75, -1]))


def main():
    with open("patronsDifficult.txt") as f:
        lines = f.readlines()

    phis = []
    expected_outputs = []
    i = 0

    for line in lines:
        numbers = map(lambda s: float(s), line.split())
        _len = len(numbers)
        phis.append(numbers[0:_len-1])
        phis[i].append(-1.0)
        expected_outputs.append(numbers[_len-1])
        i += 1
    #print(phis)
    #print(expected_outputs)
    learn_patterns(phis, expected_outputs)


if __name__ == "__main__":
    main()
