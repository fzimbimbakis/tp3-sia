from multilayer_perceptron import MultilayerPerceptron
from fonts import get_fonts, font_header
import numpy as np
import matplotlib.pyplot as plt


def batch_func(learning_rate, epochs):
    fonts = get_fonts()
    layers = np.array([len(fonts[0]), 20, 10, 2, 10, 20, len(fonts[0])])
    perceptron = MultilayerPerceptron(fonts, fonts, learning_rate, layers, batch_size=fonts.size)

    perceptron.train(epochs)
