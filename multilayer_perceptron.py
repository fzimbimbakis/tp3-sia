import matplotlib.pyplot as plt
import numpy as np
from fonts import add_noise


def sigmoid(x):
    return np.tanh(0.7*x)


def sigmoid_derivative(x):
    return (1 - np.power(x, 2))*0.7


class MultilayerPerceptron:

    def __init__(self, n_of_inputs, hidden_layers, n_of_outputs, max_error):
        self.max_error = max_error
        self.n_of_inputs = n_of_inputs
        self.hidden_layers = hidden_layers
        self.n_of_outputs = n_of_outputs

        # Array con la cantidad de neuronas de cada capa
        neurons_of_layer = [n_of_inputs] + hidden_layers + [n_of_outputs]

        weights = []
        biases = []
        last_deltas = []
        last_derivates = []
        derivatives = []
        deltas = []
        for i in range(len(neurons_of_layer) - 1):
            w = np.random.normal(loc=0.0, scale=np.sqrt(2 / (neurons_of_layer[i] + neurons_of_layer[i + 1])),
                                 size=(neurons_of_layer[i], neurons_of_layer[i + 1]))
            b = np.random.rand(neurons_of_layer[i + 1], 1)
            d = np.zeros((neurons_of_layer[i], neurons_of_layer[i + 1]))
            deltas_i = np.zeros((neurons_of_layer[i + 1], 1))
            deltas.append(deltas_i)
            last_deltas.append(deltas_i)
            last_derivates.append(d)
            derivatives.append(d)
            weights.append(w)
            biases.append(b)
        self.weights = weights
        self.biases = biases
        self.derivatives = derivatives
        self.deltas = deltas
        self.last_deltas = deltas
        self.last_derivatives = last_derivates
        activations = []
        for i in range(len(neurons_of_layer)):
            a = np.zeros(neurons_of_layer[i])
            activations.append(a)
        self.activations = activations


    def propagate(self, input_):
        return self.predict_from_layer(input_, 0)

    def predict_from_layer(self, input_, layer):
        self.activations[0] = input_

        for i, w in enumerate(self.weights):

            if i >= layer:
                x = np.dot(input_, w) + self.biases[i].T
                x = x.reshape(x.shape[1])
                input_ = sigmoid(x)
                self.activations[i + 1] = input_

        return self.activations[-1], self.weights

    def train(self, inputs, outputs, epochs, eta, K, a, Q, b, adaptive_lr=False):
        loss = []
        x = []
        etas = []
        min_weights = []
        min_loss = 100000
        dec_loss = 0
        gr_loss = 0
        loss_value = 0

        for i in range(epochs):
            total_error = 0
            for j, input_ in enumerate(inputs):
                predicted_output, weights = self.propagate(input_)

                error = outputs[j] - predicted_output

                self.backpropagation(error)

                self.update_weights(eta)

                total_error += self.mean_square_error(outputs[j], predicted_output)
            last_loss = loss_value
            loss_value = total_error / len(inputs)
            if loss_value < min_loss:
                min_loss = loss_value
                min_weights = weights
            if (i+1) % (epochs/10) == 0:
                print("Loss: {} at epoch {}".format(loss_value, i + 1))
            if loss_value <= self.max_error:
                break
            if (loss_value - last_loss) <= 0:
                gr_loss += 1
                dec_loss = 0
            else:
                dec_loss += 1
                gr_loss = 0
            if gr_loss >= K:
                eta += a * eta
                gr_loss = 0
            elif dec_loss >= Q:
                eta -= b * eta
                dec_loss = 0
            if adaptive_lr:
                etas.append(eta)
                eta = self.exp_decay(i, etas[0])
            loss.append(loss_value)
            x.append(i)
        print("Minimum loss: ", min_loss)
        self.weights = min_weights
        plt.plot(x, loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def train_with_noise(self, inputs, outputs, epochs, eta, noise, K, a, Q, b, adaptive_lr=False):
        loss = []
        x = []
        etas = []
        min_weights = []
        min_loss = 100000
        dec_loss = 0
        gr_loss = 0
        loss_value = 0

        for i in range(epochs):
            if not i % 100:
                inputs = add_noise(inputs, noise)
            total_error = 0
            for j, input_ in enumerate(inputs):
                predicted_output, weights = self.propagate(input_)

                error = outputs[j] - predicted_output

                self.backpropagation(error)

                self.update_weights(eta)

                total_error += self.mean_square_error(outputs[j], predicted_output)
            last_loss = loss_value
            loss_value = total_error / len(inputs)
            if loss_value < min_loss:
                min_loss = loss_value
                min_weights = weights
            if (i+1) % (epochs/10) == 0:
                print("Loss: {} at epoch {}".format(loss_value, i + 1))
            if loss_value <= self.max_error:
                break
            if (loss_value - last_loss) <= 0:
                gr_loss += 1
                dec_loss = 0
            else:
                dec_loss += 1
                gr_loss = 0
            if gr_loss >= K:
                eta += a * eta
                gr_loss = 0
            elif dec_loss >= Q:
                eta -= b * eta
                dec_loss = 0
            if adaptive_lr:
                etas.append(eta)
                eta = self.exp_decay(i, etas[0])
            loss.append(loss_value)
            x.append(i)
        print("Minimum loss: ", min_loss)
        self.weights = min_weights
        plt.plot(x, loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()

    def backpropagation(self, error):
        for i in reversed(range(len(self.derivatives))):
            output = self.activations[i + 1]

            delta = sigmoid_derivative(output) * error
            self.last_deltas[i] = self.deltas[i]
            self.deltas[i] = delta.reshape(delta.shape[0], -1).T

            inputs = self.activations[i]
            inputs = inputs.reshape(inputs.shape[0], -1)
            self.last_derivatives[i] = self.derivatives[i]
            self.derivatives[i] = np.dot(inputs, self.deltas[i])

            error = np.dot(self.deltas[i], self.weights[i].T)
            error = error.reshape(error.shape[1])

    def update_weights(self, eta):

        for i in range(len(self.weights)):
            self.weights[i] += eta * self.derivatives[i] + (eta * 0.9) * self.last_derivatives[i]
            self.biases[i] += eta * self.deltas[i].reshape(self.biases[i].shape) + (eta * 0.9) * self.last_deltas[i].reshape(self.biases[i].shape)

    def mean_square_error(self, expected, predicted_output):
        return np.average((expected - predicted_output) ** 2)

    def exp_decay(self, epoch, eta):
        k = 0.000001
        x = np.exp(-k * epoch)
        return eta * x


