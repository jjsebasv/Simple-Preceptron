import random

def get_weights(n):
    return [random.uniform(-0.5, 0.5) for i in range(n)]

def dot(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors should have the same length")

    weighted_outputs = [x * y for x, y in zip(v1, v2)]
    return sum(weighted_outputs)

def get_output(phi, weights, f):
    return f(dot(phi, weights))

def learn_pattern(phi, expected_output, weights, f, etha):
    output = get_output(phi, weights, f)
    correction = [val * etha * (expected_output - output) for val in phi]
    return [w + c for w, c in zip(weights, correction)]

def learn_patterns(phis, expected_outputs):
    N = 100000
    etha = 0.05
    weights = get_weights(len(phis[0]))

    def f(value):
        return 1 if value >= 0 else -1

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
    for line in lines:
        numbers = [float(s) for s in line.split()]
        input_parameters = numbers[0:-1]
        expected_output = numbers[-1]
        phis.append(input_parameters + [-1.0])
        expected_outputs.append(expected_output)
    #print(phis)
    #print(expected_outputs)
    learn_patterns(phis, expected_outputs)

if __name__ == "__main__":
    main()
