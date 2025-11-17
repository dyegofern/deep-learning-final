"""
Advanced Generative Models for Tabular Data Augmentation
CSCA 5642 - Final Project Enhancement
University of Colorado Boulder

This module implements advanced architectures:
1. Variational Autoencoder (VAE)
2. CTGAN-VAE Hybrid
3. Bidirectional GAN (BiGAN)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Optional


class TabularVAE(keras.Model):
    """
    Variational Autoencoder for Tabular Data.

    VAEs learn a latent representation of the data and can generate
    new samples by sampling from the learned distribution.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 64,
                 encoder_dims: List[int] = [256, 128],
                 decoder_dims: List[int] = [128, 256],
                 beta: float = 1.0,
                 name: str = 'tabular_vae'):
        """
        Initialize TabularVAE.

        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        latent_dim : int
            Dimension of latent space
        encoder_dims : list
            Hidden layer dimensions for encoder
        decoder_dims : list
            Hidden layer dimensions for decoder
        beta : float
            Weight for KL divergence term (beta-VAE)
        name : str
            Model name
        """
        super(TabularVAE, self).__init__(name=name)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Build encoder
        self.encoder = self._build_encoder(input_dim, latent_dim, encoder_dims)

        # Build decoder
        self.decoder = self._build_decoder(latent_dim, input_dim, decoder_dims)

    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        """Build encoder network."""
        inputs = keras.Input(shape=(input_dim,), name='encoder_input')
        x = inputs

        # Hidden layers
        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'encoder_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'encoder_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'encoder_dropout_{idx}')(x)

        # Latent space parameters
        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

        # Sampling layer
        z = layers.Lambda(self._sampling, name='z')([z_mean, z_log_var])

        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def _build_decoder(self, latent_dim, output_dim, hidden_dims):
        """Build decoder network."""
        latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
        x = latent_inputs

        # Hidden layers
        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'decoder_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'decoder_dropout_{idx}')(x)

        # Output layer (tanh for normalized data)
        outputs = layers.Dense(output_dim, activation='tanh', name='decoder_output')(x)

        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        return decoder

    def _sampling(self, args):
        """Reparameterization trick: sample from N(z_mean, exp(z_log_var))."""
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        """Forward pass through VAE."""
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstructed = self.decoder(z, training=training)
        return reconstructed, z_mean, z_log_var

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples from latent space."""
        z_samples = tf.random.normal((n_samples, self.latent_dim))
        generated = self.decoder(z_samples, training=False)
        return generated.numpy()


class CTGANVAE(keras.Model):
    """
    Hybrid CTGAN-VAE Architecture.

    Combines the adversarial training of CTGAN with the
    structured latent space of VAE for improved generation quality.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 64,
                 noise_dim: int = 100,
                 encoder_dims: List[int] = [256, 128],
                 decoder_dims: List[int] = [128, 256],
                 discriminator_dims: List[int] = [256, 256],
                 beta: float = 0.5,
                 name: str = 'ctgan_vae'):
        """
        Initialize CTGAN-VAE hybrid.

        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        latent_dim : int
            Dimension of latent space
        noise_dim : int
            Dimension of noise for generator
        encoder_dims : list
            Hidden dimensions for encoder
        decoder_dims : list
            Hidden dimensions for decoder/generator
        discriminator_dims : list
            Hidden dimensions for discriminator
        beta : float
            Weight for KL divergence
        name : str
            Model name
        """
        super(CTGANVAE, self).__init__(name=name)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.beta = beta

        # Build encoder (maps real data to latent space)
        self.encoder = self._build_encoder(input_dim, latent_dim, encoder_dims)

        # Build generator/decoder (maps latent + noise to data space)
        self.generator = self._build_generator(latent_dim, noise_dim, input_dim, decoder_dims)

        # Build discriminator
        self.discriminator = self._build_discriminator(input_dim, discriminator_dims)

    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        """Build encoder network."""
        inputs = keras.Input(shape=(input_dim,), name='encoder_input')
        x = inputs

        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'enc_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'enc_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'enc_dropout_{idx}')(x)

        z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

        encoder = keras.Model(inputs, [z_mean, z_log_var], name='encoder')
        return encoder

    def _build_generator(self, latent_dim, noise_dim, output_dim, hidden_dims):
        """Build generator/decoder network."""
        latent_input = keras.Input(shape=(latent_dim,), name='latent_input')
        noise_input = keras.Input(shape=(noise_dim,), name='noise_input')

        # Concatenate latent and noise
        x = layers.Concatenate(name='concat_latent_noise')([latent_input, noise_input])

        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'gen_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'gen_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'gen_dropout_{idx}')(x)

        outputs = layers.Dense(output_dim, activation='tanh', name='gen_output')(x)

        generator = keras.Model([latent_input, noise_input], outputs, name='generator')
        return generator

    def _build_discriminator(self, input_dim, hidden_dims):
        """Build discriminator network."""
        inputs = keras.Input(shape=(input_dim,), name='disc_input')
        x = inputs

        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, name=f'disc_dense_{idx}')(x)
            x = layers.LeakyReLU(0.2, name=f'disc_leaky_{idx}')(x)
            x = layers.Dropout(0.3, name=f'disc_dropout_{idx}')(x)

        outputs = layers.Dense(1, name='disc_output')(x)

        discriminator = keras.Model(inputs, outputs, name='discriminator')
        return discriminator

    def call(self, inputs, training=False):
        """Forward pass."""
        z_mean, z_log_var = self.encoder(inputs, training=training)

        # Sample from latent distribution
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Generate with noise
        noise = tf.random.normal((tf.shape(z)[0], self.noise_dim))
        reconstructed = self.generator([z, noise], training=training)

        return reconstructed, z_mean, z_log_var

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples."""
        # Sample from prior
        z_samples = tf.random.normal((n_samples, self.latent_dim))
        noise_samples = tf.random.normal((n_samples, self.noise_dim))

        generated = self.generator([z_samples, noise_samples], training=False)
        return generated.numpy()


class BiGAN(keras.Model):
    """
    Bidirectional Generative Adversarial Network.

    BiGAN learns both the generative and inference mappings simultaneously:
    - Generator: z -> x (latent to data)
    - Encoder: x -> z (data to latent)
    - Discriminator: discriminates (x, z) pairs
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 64,
                 encoder_dims: List[int] = [256, 128],
                 generator_dims: List[int] = [128, 256],
                 discriminator_dims: List[int] = [256, 256],
                 name: str = 'bigan'):
        """
        Initialize BiGAN.

        Parameters:
        -----------
        input_dim : int
            Dimension of input data
        latent_dim : int
            Dimension of latent space
        encoder_dims : list
            Hidden dimensions for encoder
        generator_dims : list
            Hidden dimensions for generator
        discriminator_dims : list
            Hidden dimensions for discriminator
        name : str
            Model name
        """
        super(BiGAN, self).__init__(name=name)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder (data -> latent)
        self.encoder = self._build_encoder(input_dim, latent_dim, encoder_dims)

        # Build generator (latent -> data)
        self.generator = self._build_generator(latent_dim, input_dim, generator_dims)

        # Build discriminator (discriminates (x, z) pairs)
        self.discriminator = self._build_discriminator(input_dim, latent_dim, discriminator_dims)

    def _build_encoder(self, input_dim, latent_dim, hidden_dims):
        """Build encoder network."""
        inputs = keras.Input(shape=(input_dim,), name='encoder_input')
        x = inputs

        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'enc_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'enc_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'enc_dropout_{idx}')(x)

        z = layers.Dense(latent_dim, activation='tanh', name='z_output')(x)

        encoder = keras.Model(inputs, z, name='encoder')
        return encoder

    def _build_generator(self, latent_dim, output_dim, hidden_dims):
        """Build generator network."""
        z_input = keras.Input(shape=(latent_dim,), name='z_input')
        x = z_input

        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'gen_dense_{idx}')(x)
            x = layers.BatchNormalization(name=f'gen_bn_{idx}')(x)
            x = layers.Dropout(0.2, name=f'gen_dropout_{idx}')(x)

        x_output = layers.Dense(output_dim, activation='tanh', name='x_output')(x)

        generator = keras.Model(z_input, x_output, name='generator')
        return generator

    def _build_discriminator(self, data_dim, latent_dim, hidden_dims):
        """Build discriminator for (x, z) pairs."""
        x_input = keras.Input(shape=(data_dim,), name='x_input')
        z_input = keras.Input(shape=(latent_dim,), name='z_input')

        # Concatenate x and z
        combined = layers.Concatenate(name='concat_xz')([x_input, z_input])

        x = combined
        for idx, dim in enumerate(hidden_dims):
            x = layers.Dense(dim, name=f'disc_dense_{idx}')(x)
            x = layers.LeakyReLU(0.2, name=f'disc_leaky_{idx}')(x)
            x = layers.Dropout(0.3, name=f'disc_dropout_{idx}')(x)

        validity = layers.Dense(1, activation='sigmoid', name='validity')(x)

        discriminator = keras.Model([x_input, z_input], validity, name='discriminator')
        return discriminator

    def call(self, inputs, training=False):
        """Forward pass."""
        # Encode real data
        z_encoded = self.encoder(inputs, training=training)

        # Generate fake data from random latent
        z_random = tf.random.normal((tf.shape(inputs)[0], self.latent_dim))
        x_generated = self.generator(z_random, training=training)

        return x_generated, z_encoded, z_random

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples."""
        z_samples = tf.random.normal((n_samples, self.latent_dim))
        generated = self.generator(z_samples, training=False)
        return generated.numpy()


def build_vae(input_dim: int,
              latent_dim: int = 64,
              encoder_dims: List[int] = [256, 128],
              decoder_dims: List[int] = [128, 256],
              beta: float = 1.0) -> TabularVAE:
    """
    Build and return a TabularVAE model.

    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    latent_dim : int
        Dimension of latent space
    encoder_dims : list
        Encoder hidden dimensions
    decoder_dims : list
        Decoder hidden dimensions
    beta : float
        KL divergence weight

    Returns:
    --------
    TabularVAE : Initialized VAE model
    """
    vae = TabularVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
        beta=beta
    )

    print("TabularVAE model built successfully!")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Beta (KL weight): {beta}")

    return vae


def build_ctgan_vae(input_dim: int,
                    latent_dim: int = 64,
                    noise_dim: int = 100,
                    beta: float = 0.5) -> CTGANVAE:
    """
    Build and return a CTGAN-VAE hybrid model.

    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    latent_dim : int
        Dimension of latent space
    noise_dim : int
        Dimension of noise
    beta : float
        KL divergence weight

    Returns:
    --------
    CTGANVAE : Initialized hybrid model
    """
    model = CTGANVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        noise_dim=noise_dim,
        beta=beta
    )

    print("CTGAN-VAE hybrid model built successfully!")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Noise dimension: {noise_dim}")

    return model


def build_bigan(input_dim: int,
                latent_dim: int = 64) -> BiGAN:
    """
    Build and return a BiGAN model.

    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    latent_dim : int
        Dimension of latent space

    Returns:
    --------
    BiGAN : Initialized BiGAN model
    """
    bigan = BiGAN(
        input_dim=input_dim,
        latent_dim=latent_dim
    )

    print("BiGAN model built successfully!")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")

    return bigan


if __name__ == '__main__':
    print("Advanced generative models loaded successfully!")
    print("\nAvailable models:")
    print("  - TabularVAE")
    print("  - CTGANVAE")
    print("  - BiGAN")
