from multilayer_perceptron import MultilayerPerceptron
from fonts import get_fonts, print_letter
import numpy as np

# BATCH y momentum

fonts = get_fonts()
layers = np.array([fonts[0].size, 20, 10, 2, 10, 20, fonts[0].size])
learning_rate = 0.0005
epochs = 20000

perceptron = MultilayerPerceptron(fonts, fonts, learning_rate, layers, batch_size=fonts[0].size, momentum=True)

perceptron.train(epochs)

output = perceptron.test_input(fonts)
for i in output:
    print_letter(i)

for i in fonts:
    print_letter(i)

print(fonts[0])
