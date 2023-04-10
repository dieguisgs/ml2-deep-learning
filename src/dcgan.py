import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time
from sys import platform
from tqdm import trange
from typing import Tuple

if platform == "darwin":
    # Fix macOS error "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized."
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DCGAN:
    """Implementation of a Deep Convolutional Generative Adversarial Network.

    Adapted from: https://github.com/eriklindernoren/Keras-GAN

    Examples:
        1. Training the network. Optionally, save sample images.
            dcgan = DCGAN(dataset='mnist')
            dcgan.train(epochs=4000, batch_size=32, save_interval=50)

    """

    def __init__(self, dataset: str = 'mnist'):
        """Deep convolutional GAN class initializer.

        Args:
            dataset: Sample dataset { mnist, fashion_mnist }.

        """
        self._dataset = self._load_dataset(dataset)

        # Input shape
        self._channels = self._dataset.shape[3]
        self._img_shape = (self._dataset.shape[1], self._dataset.shape[2], self._channels)
        self._latent_dim = 100  # Random vector noise size. Experimental de facto standard.

        optimizer = tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        print('\n\nMODEL SUMMARY')

        # Build and compile the discriminator
        self._discriminator = self._build_discriminator()
        self._discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self._generator = self._build_generator()

        # The generator takes an array of random elements (noise) as input and generates images
        noise = tf.keras.Input(shape=(self._latent_dim,))  # The comma is necessary when you have only one dimension
        images = self._generator(noise)

        # For the combined model we will only train the generator
        self._discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self._discriminator(images)

        # The combined model (stacked generator and discriminator)
        # trains the generator to fool the discriminator
        self._combined = tf.keras.Model(inputs=noise, outputs=validity)
        self._combined.layers[0]._name = 'noise'
        self._combined.layers[1]._name = 'generator'
        self._combined.layers[2]._name = 'discriminator'

        print('\n')
        self._combined._name = 'Generator_Discriminator'
        self._combined.summary()

        # Compile the combined model
        self._combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, epochs: int = 1000, batch_size: int = 128, save_interval: int = 50, images_dir: str = "images/tf_mnist", models_dir: str = "models"):
        """Train the discriminator and the generator.

        Saves both models upon completion.

        Args:
        -------
        epochs: 
            Number of times the dataset is passed forward and backward through the neural network.
        batch_size: 
            Number of examples used in one iteration.
        save_interval: 
            Save interval for sample generated images [epochs]. Set to 0 to disable.

        """
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print("\n\nTraining...")

        # Create progress bar
        pbar = trange(1, epochs + 1, total=epochs, unit="epoch", file=sys.stdout)

        # Train
        for epoch in pbar:
            # DISCRIMINATOR
            # Select a random half of images
            idx = np.random.randint(low=0, high=self._dataset.shape[0], size=batch_size)
            real_images = self._dataset[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
            fake_images = self._generator.predict(noise)

            # Train the discriminator (real images classified as ones and fake images as zeros)
            discriminator_loss_real = self._discriminator.train_on_batch(real_images, valid)
            discriminator_loss_fake = self._discriminator.train_on_batch(fake_images, fake)
            discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

            # GENERATOR
            # Train the generator to deceive the discriminator (make it believe fake images are real)
            generator_loss = self._combined.train_on_batch(noise, valid)

            # Print progress
            pbar.set_description("[Discriminator loss: %.4f, accuracy: %.2f%%] - [Generator loss: %.4f]"
                                 % (discriminator_loss[0], 100 * discriminator_loss[1], generator_loss))
            pbar.update(1)
            time.sleep(0.01)  # Prevents a race condition between tqdm and print statements.

            # Save generated image samples
            if save_interval > 0 and epoch % save_interval == 0:
                self._generate_samples(epoch, images_dir)

        # Save final models
        os.makedirs('models', exist_ok=True)

        self._generator.save(os.path.join(models_dir, f"{images_dir.split('/')[-1]}_generator.h5"))
        self._discriminator.save(os.path.join(models_dir, f"{images_dir.split('/')[-1]}_discriminator.h5"))

    @staticmethod
    def _load_dataset(dataset: str):
        """Loads a sample dataset and rescales images to [-1, 1].

        Args:
            dataset: Sample dataset { mnist, fashion_mnist }.

        Raises:
            ValueError: If the dataset is unknown.

        Returns:
            Normalized training set images.

        """
        if dataset == 'mnist':
            (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        elif dataset == 'fashion_mnist':
            (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            raise ValueError("Dataset not supported. Possible values are 'mnist' and 'fashion_mnist'.")

        # Rescale images to [-1, 1]
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        return x_train

    def _build_discriminator(self):
        """Build the discriminator network.

        Returns:
            A TensorFlow Keras model.

        """
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)

        model = tf.keras.Sequential([
            # Parameters = (kernel_height * kernel_width * channels + bias) * filters = (5 * 5 * 1 + 1) * 64 = 1664
            tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, input_shape=self._img_shape, padding='same', kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),  # Randomly zero 30% of inputs (cells) at each training update to prevent overfitting

            # Parameters = (kernel_height * kernel_width * channels + bias) * filters = (5 * 5 * 64 + 1) * 128 = 204928
            tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', kernel_initializer=kernel_initializer),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Parameters = (Flatten + bias) * outputs = ((7 * 7 * 128) + 1) * 1 = 6273
        ])

        print('\n')
        model._name = 'Discriminator'
        model.summary()

        image = tf.keras.Input(shape=self._img_shape)
        validity = model(image)

        return tf.keras.Model(inputs=image, outputs=validity)

    def _build_generator(self):
        """Build the generator network.

        Returns:
            A TensorFlow Keras model.

        """
        kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.02)

        model = tf.keras.Sequential([
            # Parameters = (7 * 7 * 128) * (input_dim + bias) = (7 * 7 * 128) * (100 + 1) = 633472
            tf.keras.layers.Dense(7 * 7 * 128, input_dim=self._latent_dim, kernel_initializer=kernel_initializer),

            # BatchNormalization: Normalize layer inputs (mu = 0, sigma = 1).
            #     mu_moving = momentum * mu_moving + (1 - momentum) * mu_batch
            #     sigma_moving = momentum * sigma_moving + (1 - momentum) * sigma_batch
            tf.keras.layers.BatchNormalization(momentum=0.9),  # Parameters = 4 * channels = 4 * 6272 = 25088
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Reshape((7, 7, 128)),  # (height, width, channels)

            # Parameters = (kernel_height * kernel_width * channels + bias) * filters = (3 * 3 * 128 + 1) * 128 = 147584
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(momentum=0.9),  # Parameters = 4 * channels = 4 * 128 = 512
            tf.keras.layers.LeakyReLU(alpha=0.2),

            # Parameters = (kernel_height * kernel_width * channels + bias) * filters = (3 * 3 * 128 + 1) * 64 = 73792
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(momentum=0.9),  # Parameters = 4 * channels = 4 * 64 = 256
            tf.keras.layers.LeakyReLU(alpha=0.2),

            # Parameters = (kernel_height * kernel_width * channels + bias) * filters = (3 * 3 * 64 + 1) * 1 = 577
            tf.keras.layers.Conv2D(filters=self._channels, kernel_size=3, padding='same', kernel_initializer=kernel_initializer),

            # Because we will be normalizing training inputs to lie in the range [-1, 1], the tanh function
            # should be used for the last layer of the generator to ensure its output is also within this range.
            tf.keras.layers.Activation('tanh')
        ])

        print('\n')
        model._name = 'Generator'
        model.summary()  # Output shape format: (batch_size, height, width, channels)

        noise = tf.keras.Input(shape=(self._latent_dim,))
        image = model(noise)

        return tf.keras.Model(inputs=noise, outputs=image)

    def _generate_samples(self, epoch: int, images_dir: str, figure_size: Tuple[int, int] = (10, 10)):
        """Saves a .png figure composed of several generated images arranged as specified in the figure_size parameter.

        Args:
            epoch: Training epoch number. Used to name the saved image.
            figure_size: (rows, columns) layout of the output image.

        """
        # Create folder if it does not exist
        os.makedirs(images_dir, exist_ok=True)
        # Create folder if it does not exist

        # Generate row * cols images
        rows, cols = figure_size
        noise = np.random.normal(0, 1, size=(rows * cols, self._latent_dim))
        generated_images = self._generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]

        # Build and save a single figure with all the generated images side-by-side
        plt.figure(figsize=figure_size)

        for i in range(generated_images.shape[0]):
            plt.subplot(rows, cols, i+1)
            plt.imshow(generated_images[i, :, :, 0], interpolation='nearest', cmap='gray_r')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "epoch_%04d.png" % epoch))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN('fashion_mnist')
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)