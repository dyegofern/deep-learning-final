"""
CTGAN (Conditional Tabular GAN) Architecture for Aviation Emissions Data
CSCA 5642 - Final Project
University of Colorado Boulder

This module implements a Conditional Tabular GAN using Keras/TensorFlow
for generating synthetic aviation emissions data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List


class CTGANGenerator(keras.Model):
    """
    CTGAN Generator Network.

    Generates synthetic tabular data conditioned on categorical features.
    Architecture: Noise + Condition → Dense layers with BatchNorm → Synthetic data
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 condition_dim: int = 0,
                 hidden_dims: List[int] = [256, 256],
                 dropout_rate: float = 0.2,
                 name: str = 'ctgan_generator'):
        """
        Initialize CTGAN Generator.

        Parameters:
        -----------
        input_dim : int
            Dimension of random noise vector
        output_dim : int
            Dimension of generated data (number of features)
        condition_dim : int
            Dimension of conditional input (e.g., one-hot encoded categories)
        hidden_dims : list
            List of hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        name : str
            Model name
        """
        super(CTGANGenerator, self).__init__(name=name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.total_input_dim = input_dim + condition_dim

        # Build generator layers
        self.dense_layers = []
        self.batch_norm_layers = []
        self.dropout_layers = []

        # Hidden layers
        for idx, hidden_dim in enumerate(hidden_dims):
            self.dense_layers.append(
                layers.Dense(hidden_dim, name=f'dense_{idx}')
            )
            self.batch_norm_layers.append(
                layers.BatchNormalization(name=f'batch_norm_{idx}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{idx}')
            )

        # Output layer
        self.output_layer = layers.Dense(output_dim, activation='tanh', name='output')

    def call(self, inputs, training=False):
        """
        Forward pass through generator.

        Parameters:
        -----------
        inputs : tuple or tensor
            If condition_dim > 0: (noise, condition)
            Otherwise: noise
        training : bool
            Whether in training mode

        Returns:
        --------
        tensor : Generated synthetic data
        """
        # Handle conditional vs unconditional input
        if self.condition_dim > 0:
            noise, condition = inputs
            x = tf.concat([noise, condition], axis=-1)
        else:
            x = inputs

        # Forward through hidden layers
        for dense, batch_norm, dropout in zip(self.dense_layers,
                                               self.batch_norm_layers,
                                               self.dropout_layers):
            x = dense(x)
            x = batch_norm(x, training=training)
            x = tf.nn.relu(x)
            x = dropout(x, training=training)

        # Output layer
        output = self.output_layer(x)

        return output

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'condition_dim': self.condition_dim
        }


class CTGANDiscriminator(keras.Model):
    """
    CTGAN Discriminator (Critic) Network.

    Classifies data as real or synthetic using Wasserstein distance.
    Architecture: Data (+ Condition) → Dense layers with LeakyReLU → Wasserstein score
    """

    def __init__(self,
                 input_dim: int,
                 condition_dim: int = 0,
                 hidden_dims: List[int] = [256, 256],
                 dropout_rate: float = 0.2,
                 name: str = 'ctgan_discriminator'):
        """
        Initialize CTGAN Discriminator.

        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        condition_dim : int
            Dimension of conditional input
        hidden_dims : list
            List of hidden layer dimensions
        dropout_rate : float
            Dropout rate for regularization
        name : str
            Model name
        """
        super(CTGANDiscriminator, self).__init__(name=name)

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.total_input_dim = input_dim + condition_dim

        # Build discriminator layers
        self.dense_layers = []
        self.dropout_layers = []

        # Hidden layers (no batch norm in discriminator for WGAN-GP)
        for idx, hidden_dim in enumerate(hidden_dims):
            self.dense_layers.append(
                layers.Dense(hidden_dim, name=f'dense_{idx}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_{idx}')
            )

        # Output layer (no activation for Wasserstein distance)
        self.output_layer = layers.Dense(1, name='output')

    def call(self, inputs, training=False):
        """
        Forward pass through discriminator.

        Parameters:
        -----------
        inputs : tuple or tensor
            If condition_dim > 0: (data, condition)
            Otherwise: data
        training : bool
            Whether in training mode

        Returns:
        --------
        tensor : Wasserstein distance score
        """
        # Handle conditional vs unconditional input
        if self.condition_dim > 0:
            data, condition = inputs
            x = tf.concat([data, condition], axis=-1)
        else:
            x = inputs

        # Forward through hidden layers
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)

        # Output layer
        output = self.output_layer(x)

        return output

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim
        }


class CTGAN:
    """
    Complete CTGAN (Conditional Tabular GAN) implementation.

    Combines Generator and Discriminator with Wasserstein loss and gradient penalty.
    """

    def __init__(self,
                 data_dim: int,
                 noise_dim: int = 100,
                 condition_dim: int = 0,
                 generator_dims: List[int] = [256, 256],
                 discriminator_dims: List[int] = [256, 256],
                 generator_lr: float = 2e-4,
                 discriminator_lr: float = 2e-4,
                 gradient_penalty_weight: float = 10.0):
        """
        Initialize CTGAN.

        Parameters:
        -----------
        data_dim : int
            Dimension of real data
        noise_dim : int
            Dimension of random noise for generator
        condition_dim : int
            Dimension of conditional input
        generator_dims : list
            Hidden layer dimensions for generator
        discriminator_dims : list
            Hidden layer dimensions for discriminator
        generator_lr : float
            Learning rate for generator optimizer
        discriminator_lr : float
            Learning rate for discriminator optimizer
        gradient_penalty_weight : float
            Weight for gradient penalty term
        """
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.gradient_penalty_weight = gradient_penalty_weight

        # Build generator and discriminator
        self.generator = CTGANGenerator(
            input_dim=noise_dim,
            output_dim=data_dim,
            condition_dim=condition_dim,
            hidden_dims=generator_dims
        )

        self.discriminator = CTGANDiscriminator(
            input_dim=data_dim,
            condition_dim=condition_dim,
            hidden_dims=discriminator_dims
        )

        # Optimizers
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=generator_lr,
            beta_1=0.5,
            beta_2=0.9
        )

        self.discriminator_optimizer = keras.optimizers.Adam(
            learning_rate=discriminator_lr,
            beta_1=0.5,
            beta_2=0.9
        )

    def gradient_penalty(self, real_data, fake_data, condition=None):
        """
        Compute gradient penalty for WGAN-GP.

        Parameters:
        -----------
        real_data : tensor
            Real data samples
        fake_data : tensor
            Generated fake data samples
        condition : tensor, optional
            Conditional input

        Returns:
        --------
        tensor : Gradient penalty value
        """
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        # Interpolated data
        interpolated = alpha * real_data + (1 - alpha) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)

            # Get discriminator output
            if condition is not None:
                disc_output = self.discriminator([interpolated, condition], training=True)
            else:
                disc_output = self.discriminator(interpolated, training=True)

        # Compute gradients
        gradients = tape.gradient(disc_output, interpolated)

        # Compute gradient norm
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))

        # Gradient penalty
        penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))

        return penalty

    @tf.function
    def train_discriminator_step(self, real_data, condition=None):
        """
        Single training step for discriminator.

        Parameters:
        -----------
        real_data : tensor
            Real data batch
        condition : tensor, optional
            Conditional input batch

        Returns:
        --------
        tuple : (discriminator_loss, wasserstein_distance, gradient_penalty)
        """
        batch_size = tf.shape(real_data)[0]

        # Generate noise
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            # Generate fake data
            if condition is not None:
                fake_data = self.generator([noise, condition], training=True)
                real_output = self.discriminator([real_data, condition], training=True)
                fake_output = self.discriminator([fake_data, condition], training=True)
            else:
                fake_data = self.generator(noise, training=True)
                real_output = self.discriminator(real_data, training=True)
                fake_output = self.discriminator(fake_data, training=True)

            # Wasserstein distance
            wasserstein_distance = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data, condition)

            # Total discriminator loss
            disc_loss = wasserstein_distance + self.gradient_penalty_weight * gp

        # Update discriminator
        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )

        return disc_loss, wasserstein_distance, gp

    @tf.function
    def train_generator_step(self, batch_size, condition=None):
        """
        Single training step for generator.

        Parameters:
        -----------
        batch_size : int
            Batch size
        condition : tensor, optional
            Conditional input batch

        Returns:
        --------
        tensor : Generator loss
        """
        # Generate noise
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            # Generate fake data
            if condition is not None:
                fake_data = self.generator([noise, condition], training=True)
                fake_output = self.discriminator([fake_data, condition], training=True)
            else:
                fake_data = self.generator(noise, training=True)
                fake_output = self.discriminator(fake_data, training=True)

            # Generator loss (maximize discriminator output for fake data)
            gen_loss = -tf.reduce_mean(fake_output)

        # Update generator
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        return gen_loss

    def generate_samples(self, n_samples: int, condition=None) -> np.ndarray:
        """
        Generate synthetic samples.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        condition : array-like, optional
            Conditional input for generation

        Returns:
        --------
        array : Generated synthetic data
        """
        noise = tf.random.normal([n_samples, self.noise_dim])

        if condition is not None:
            synthetic_data = self.generator([noise, condition], training=False)
        else:
            synthetic_data = self.generator(noise, training=False)

        return synthetic_data.numpy()


def build_ctgan(data_dim: int,
                noise_dim: int = 100,
                condition_dim: int = 0,
                generator_lr: float = 2e-4,
                discriminator_lr: float = 2e-4) -> CTGAN:
    """
    Build and return a CTGAN model.

    Parameters:
    -----------
    data_dim : int
        Dimension of real data
    noise_dim : int
        Dimension of random noise
    condition_dim : int
        Dimension of conditional input
    generator_lr : float
        Generator learning rate
    discriminator_lr : float
        Discriminator learning rate

    Returns:
    --------
    CTGAN : Initialized CTGAN model
    """
    ctgan = CTGAN(
        data_dim=data_dim,
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr
    )

    print("CTGAN model built successfully!")
    print(f"  Data dimension: {data_dim}")
    print(f"  Noise dimension: {noise_dim}")
    print(f"  Condition dimension: {condition_dim}")

    return ctgan


if __name__ == '__main__':
    print("CTGAN model classes loaded successfully!")
    print("\nAvailable classes:")
    print("  - CTGANGenerator")
    print("  - CTGANDiscriminator")
    print("  - CTGAN (complete model)")
    print("\nExample usage:")
    print("  ctgan = build_ctgan(data_dim=30, noise_dim=100, condition_dim=10)")
