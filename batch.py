from multilayer_perceptron import MultilayerPerceptron
import fonts
import numpy as np
import matplotlib.pyplot as plt

fonts = fonts.get_fonts()
layers = np.array([len(fonts[0]), 20, 10, 2, 10, 20, len(fonts[0])])
learning_rate = 0.0005
epochs = 20000

perceptron = MultilayerPerceptron(fonts, fonts, learning_rate, layers, batch_size=fonts.size)

perceptron.train(epochs)
