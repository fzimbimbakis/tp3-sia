from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.datasets import mnist
import json

from Exercise2.plot import plotter

# import tensorflow as tf

with open("../config.json", "r") as jsonfile:
    jsonData = json.load(jsonfile)  # Reading the file
    print("Read successful")
    jsonfile.close()

data = jsonData['Exercise2']
original_dim = data["original_dim"]
intermediate_dim = data["intermediate_dim"]
latent_dim = data["latent_dim"]
batch_size = data["batch_size"]
epochs = data["epochs"]
epsilon_std = data["epochs_std"]

norm = 0.1

# https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/
# Negative log likelihood of Bernoulli function
def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


# Identity transform layer that adds KL divergence to the final model loss.
class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

# https://keras.io/examples/generative/vae/
# class Sampling(Layer):
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Define the decoder
decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation="relu"),
    Dense(original_dim, activation="sigmoid")
])

# Input layer
x = Input(shape=(original_dim,))

# Hidden layer
h = Dense(intermediate_dim, activation='relu')(x)

# Output layer for mean and log variance
z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

# Normalize log variance to std dev
z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

# specify and compile the model
vae = Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer='rmsprop', loss=nll)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, original_dim) / 255.
x_test = x_test.reshape(-1, original_dim) / 255.

vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

encoder = Model(x, z_mu)

plotter(encoder, decoder, norm, x_test, y_test, batch_size)