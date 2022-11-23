import numpy as np
from layer import Layer
from fonts import add_noise
from activations import Activation

FIRST = 0
MIDDLE = 1
LAST = 2

class MultilayerPerceptron:
    adaptive_rate = False
    error_limit = 0.001
    prev_layer_neurons = 0

    def __init__(self, training_set, expected_output, learning_rate, layers, learning_rate_params=None,
                 batch_size=1, momentum=False):
        # Training set example: [[1, 1], [-1, 1], [1, -1]]
        self.training_set = training_set
        # Expected output example: [[0, 0], [0, 1], [1, 0]]
        self.expected_output = expected_output
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = []
        self.error_min = None
        self.momentum = momentum
        if learning_rate_params:
            self.adaptive_rate = True
            self.learning_rate_inc = learning_rate_params[0]
            self.learning_rate_dec = learning_rate_params[1]
            self.learning_rate_k = learning_rate_params[2]
        self.add(layers[0], FIRST)
        for i in range(layers.size-2):
            self.add(layers[i+1], MIDDLE)
        self.add(layers[-1], LAST)



    def train(self, epochs, noise_factor=None):
        error = 1
        prev_error = None
        self.error_min = float('inf')
        k = 0
        aux_batch = self.batch_size
        errors = []

        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("epoch: " + str(epoch))
                print("error: " + str(error))
            if noise_factor:
                aux_training_set = add_noise(self.training_set, noise_factor)
            else:
                aux_training_set = self.training_set
            aux_expected_output = self.expected_output
            while len(aux_training_set) > 0:
                i_x = np.random.randint(0, len(aux_training_set))
                training_set = aux_training_set[i_x]
                expected_output = aux_expected_output[i_x]

                aux_training_set = np.delete(aux_training_set, i_x, axis=0)
                aux_expected_output = np.delete(aux_expected_output, i_x, axis=0)

                self.propagate(training_set)
                self.backpropagation(expected_output)

                aux_batch -= 1
                self.update_weights(aux_batch)
                if aux_batch == 0:
                    aux_batch = self.batch_size

                aux_error = self.calculate_error(expected_output)
                error += aux_error

                if self.adaptive_rate and prev_error:
                    k = self.adapt_learning_rate(error - prev_error, k)
                prev_error = aux_error

            error *= 0.5
            errors.append(error)

            if error < self.error_min:
                self.error_min = error

            if error < self.error_limit:
                return

    def propagate(self, training_set):
        m = len(self.layers)
        self.layers[0].set_activations(training_set)
        for i in range(1, m):
            prev_layer = self.layers[i-1]
            self.layers[i].propagate(prev_layer)

    def encode_input(self, training_value):
        m = int(len(self.layers) / 2)
        self.layers[0].set_activations(training_value)
        for i in range(1, m+1):
            prev_layer = self.layers[i - 1]
            self.layers[i].propagate(prev_layer)
        return np.copy(self.layers[m].get_neurons_activation())

    def encode(self, training_set):
        answer = []
        for i in training_set:
            answer.append(self.encode_input(i))
        return answer

    def decode(self, training_value):
        m = int(len(self.layers) / 2)
        self.layers[m].set_activations(training_value)
        for i in range(m + 1, len(self.layers)):
            prev_layer = self.layers[i - 1]
            self.layers[i].propagate(prev_layer)
        return np.copy(self.layers[len(self.layers)-1].get_neurons_activation())

    def decode_input(self, training_set):
        answer = []
        for i in training_set:
            answer.append(self.decode_input(i))
        return answer

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].activation) ** 2
        return aux_sum


    def backpropagation(self, expected_output):
        m = len(self.layers)
        for i in range(m - 1, 0, -1):
            neurons = self.layers[i].neurons
            for j in range(len(neurons)):
                if i == m - 1:
                    neurons[j].sigma = Activation.sigmoid_dx(neurons[j].excitation) * \
                                       (expected_output[j] - neurons[j].activation)
                else:
                    upper_layer_neurons = self.layers[i + 1].neurons
                    aux_sum = 0
                    for neuron in upper_layer_neurons:
                        aux_sum += neuron.weights[j] * neuron.sigma
                    neurons[j].sigma = Activation.sigmoid_dx(neurons[j].excitation) * aux_sum

    def update_weights(self, batch_size):
        m = len(self.layers)
        for i in range(1, m):
            neurons = self.layers[i].neurons
            prev_neurons_activations = self.layers[i - 1].get_neurons_activation()
            for neuron in neurons:
                neuron.update_weights(self.learning_rate, prev_neurons_activations, self.momentum, batch_size)

    def add(self, neurons, layer):
        self.layers.append(Layer(neurons, self.prev_layer_neurons, layer))
        self.prev_layer_neurons = neurons

    def adapt_learning_rate(self, delta_error, k):
        if delta_error < 0:
            if k > 0:
                k = 0
            k -= 1
            if k == -self.learning_rate_k:
                self.learning_rate += self.learning_rate_inc
        elif delta_error > 0:
            if k < 0:
                k = 0
            k += 1
            if k == self.learning_rate_k:
                self.learning_rate -= self.learning_rate_dec * self.learning_rate
        else:
            k = 0
        return k

    def test_input(self, test_set):
        output = []
        for i in range(len(test_set)):
            self.propagate(test_set[i])
            output.append([neuron.activation for neuron in self.layers[len(self.layers) - 1].neurons])
        return output
