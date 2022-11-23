from multilayer_perceptron import MultilayerPerceptron
from fonts import get_fonts, font_header
import numpy as np
import matplotlib.pyplot as plt

fonts = get_fonts()
layers = np.array([len(fonts[0]), 20, 10, 2, 10, 20, len(fonts[0])])
learning_rate = 0.0005
epochs = 25000

perceptron = MultilayerPerceptron(fonts, fonts, learning_rate, layers)

perceptron.train(epochs)

encoded_inputs = perceptron.encode(fonts)

labels = np.array(font_header)

x, y = zip(*encoded_inputs)
plt.scatter(x, y)
for i, text in enumerate(labels):
    plt.annotate(text, (x[i], y[i]))
plt.show()
