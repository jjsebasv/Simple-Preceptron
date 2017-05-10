#!/usr/bin/python3

import random
from properties import Properties

def separateInput():
    props = Properties("config.properties")

    with open(props.filename) as f:
        lines = f.readlines()[1:]

    random.shuffle(lines)
    training_lines = round(props.training_percentage * len(lines))

    with open('training.data', 'w') as file:
        for line in lines[0:training_lines]:
            file.write(line)

    with open('test.data', 'w') as file:
        for line in lines[training_lines:]:
            file.write(line)


if __name__ == "__main__":
    separateInput()
