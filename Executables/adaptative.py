from multilayer_perceptron import MultilayerPerceptron
from fonts import get_fonts, print_letter
import numpy as np


def adaptative_func(learning_rate, epochs):
    fonts = get_fonts()
    layers = np.array([fonts[0].size, 20, 10, 2, 10, 20, fonts[0].size])
    perceptron = MultilayerPerceptron(fonts, fonts, learning_rate, layers, batch_size=fonts[0].size,
                                      learning_rate_params=[0.0005, 0.2, 20])

    perceptron.train(epochs)

    output = perceptron.test_input(fonts)
    for text, i in enumerate(fonts):
        print_letter(i)
        print(fonts[text])
        print(output[text])
