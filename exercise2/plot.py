import numpy as np
import matplotlib as plt


def plotter(encoder, decoder, norm, x_test, y_test, batch_size):
    plot_latent_space(encoder,x_test, y_test, batch_size)
    plot_digits(norm, decoder)


# display a 2D plot of the digit classes in the latent space
def plot_latent_space(encoder, x_test, y_test, batch_size):
    z_test = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test, alpha=.4, s=3 ** 2, cmap='viridis')
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Digit classes in the latent space")
    plt.show()


# display a 2D plot of the digits
def plot_digits(norm, decoder):
    n = 15  # figure with 15x15 digits
    digit_size = 28
    # linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian to produce values
    # of the latent variables z, since the prior of the latent space
    # is Gaussian
    u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n), np.linspace(0.05, 0.95, n)))
    z_grid = norm.ppf(u_grid)
    x_decoded = decoder.predict(z_grid.reshape(n * n, 2))
    x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
    plt.show()
