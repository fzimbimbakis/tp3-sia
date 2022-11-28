from typing import List
import numpy as np
from fonts import font_header, font, to_bits, labeled_scatter, add_noise
from multilayer_perceptron import MultilayerPerceptron
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys

basePath = '..'

def plot_heatmap(matrix, title, labels=None):
    # Create a dataset
    df = pd.DataFrame(matrix)
    # Default heatmap
    p = sns.heatmap(df, annot=labels, fmt='', cmap='Blues')
    p.set_title(title)
    plt.show()

module_path = os.path.abspath(os.path.join(basePath))
if module_path not in sys.path:
    sys.path.append(module_path)

training_points = to_bits(font)

layers: List[int] = [25, 15, 5, 2, 5, 15, 25]
layers.append(np.size(training_points, axis=1))
layers.insert(0, np.size(training_points, axis=1))

epoch = 10000
eta = 0.001
noise = 0.2

neural_network: MultilayerPerceptron = MultilayerPerceptron(np.size(training_points, 1), layers, np.size(training_points, 1), 1e-6)

count = 5
start = 1
training_points = training_values = to_bits(font)[start:start+count]
#training_points = training_values = to_bits(font3)

neural_network.train(training_points,training_values,epoch,eta,5,0.5,10,0.1,True)
#neural_network.train_with_noise(training_points, training_values, epoch, eta, noise, 5, 0.5, 10, 0.1, True)

count_test = 5
start_test = 1
#testing_points = testing_values = add_noise(to_bits(font3), noise)
#testing_points = testing_values = add_noise(to_bits(font3)[start_test:start_test+count_test], noise)
testing_points = testing_values = to_bits(font)[start_test:start_test+count_test]
#testing_points = testing_values = to_bits(font3)

z_values: np.ndarray = np.empty((np.size(testing_points, 0), 2))
predictions: np.ndarray = np.empty(training_points.shape)
# espacio latente
for i in range(np.size(testing_points, 0)):
    predictions[i], w = neural_network.propagate(testing_points[i])
    z_values[i] = neural_network.activations[len(layers)//2 + 1]
labeled_scatter(z_values[:, 0], z_values[:, 1], labels=font_header[start_test:start_test+count_test])
#labeled_scatter(z_values[:, 0], z_values[:, 1], labels=font3_lables)


for i in range(len(testing_points)):
    title = "Input Letter: " + font_header[start_test + i]
    plot_heatmap(testing_points[i].reshape(7, 5), title)
    title = "Predicted"
    plot_heatmap(predictions[i].reshape(7, 5), title)


#testing_points = testing_values = np.array([[0, 0], [0.5, 0.5], [-0.5, -0.5]])
#for i in range(np.size(testing_points, 0)):
#    predictions[i], w = neural_network.predict_from_layer(testing_points[i], len(layers)//2 + 1)

#for i in range(len(testing_points)):
#    title = "New character"
#    plot_heatmap(predictions[i].reshape(7,5), title)