"""
Advanced Training Functions for Generative Models
CSCA 5642 - Final Project Enhancement
University of Colorado Boulder

This module implements training loops for:
1. VAE
2. CTGAN-VAE Hybrid
3. BiGAN
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


def vae_loss(x_true, x_pred, z_mean, z_log_var, beta=1.0):
    """
    Compute VAE loss (reconstruction + KL divergence).

    Parameters:
    -----------
    x_true : tensor
        True data
    x_pred : tensor
        Reconstructed data
    z_mean : tensor
        Mean of latent distribution
    z_log_var : tensor
        Log variance of latent distribution
    beta : float
        Weight for KL divergence (beta-VAE)

    Returns:
    --------
    tuple : (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss (MSE for continuous data)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(x_true - x_pred), axis=-1)
    )

    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    )

    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def train_vae(vae_model,
              real_data: np.ndarray,
              epochs: int = 100,
              batch_size: int = 256,
              learning_rate: float = 1e-3,
              verbose: bool = True) -> Dict:
    """
    Train Variational Autoencoder.

    Parameters:
    -----------
    vae_model : TabularVAE
        VAE model to train
    real_data : array
        Real training data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    verbose : bool
        Print training progress

    Returns:
    --------
    dict : Training history
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(real_data)
    dataset = dataset.shuffle(buffer_size=len(real_data))
    dataset = dataset.batch(batch_size)

    # Training history
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_loss': []
    }

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            # Forward pass
            x_reconstructed, z_mean, z_log_var = vae_model(batch, training=True)

            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                batch, x_reconstructed, z_mean, z_log_var, vae_model.beta
            )

        # Backward pass
        gradients = tape.gradient(total_loss, vae_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))

        return total_loss, recon_loss, kl_loss

    # Training loop
    iterator = tqdm(range(epochs), desc='Training VAE') if verbose else range(epochs)

    for epoch in iterator:
        epoch_total_loss = []
        epoch_recon_loss = []
        epoch_kl_loss = []

        for batch in dataset:
            total_loss, recon_loss, kl_loss = train_step(batch)

            epoch_total_loss.append(total_loss.numpy())
            epoch_recon_loss.append(recon_loss.numpy())
            epoch_kl_loss.append(kl_loss.numpy())

        # Record epoch metrics
        avg_total_loss = np.mean(epoch_total_loss)
        avg_recon_loss = np.mean(epoch_recon_loss)
        avg_kl_loss = np.mean(epoch_kl_loss)

        history['total_loss'].append(avg_total_loss)
        history['reconstruction_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)

        if verbose:
            iterator.set_postfix({
                'total_loss': f'{avg_total_loss:.4f}',
                'recon_loss': f'{avg_recon_loss:.4f}',
                'kl_loss': f'{avg_kl_loss:.4f}'
            })

    return history


def train_ctgan_vae(ctgan_vae_model,
                    real_data: np.ndarray,
                    epochs: int = 100,
                    batch_size: int = 256,
                    n_critic: int = 5,
                    generator_lr: float = 2e-4,
                    discriminator_lr: float = 2e-4,
                    gradient_penalty_weight: float = 10.0,
                    verbose: bool = True) -> Dict:
    """
    Train CTGAN-VAE hybrid model.

    Parameters:
    -----------
    ctgan_vae_model : CTGANVAE
        Hybrid model to train
    real_data : array
        Real training data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    n_critic : int
        Number of discriminator updates per generator update
    generator_lr : float
        Generator learning rate
    discriminator_lr : float
        Discriminator learning rate
    gradient_penalty_weight : float
        Weight for gradient penalty
    verbose : bool
        Print training progress

    Returns:
    --------
    dict : Training history
    """
    gen_optimizer = tf.keras.optimizers.Adam(generator_lr, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(discriminator_lr, beta_1=0.5, beta_2=0.9)

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(real_data)
    dataset = dataset.shuffle(buffer_size=len(real_data))
    dataset = dataset.batch(batch_size)

    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'w_distance': [],
        'gp': [],
        'recon_loss': [],
        'kl_loss': []
    }

    @tf.function
    def gradient_penalty(real_data, fake_data):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = ctgan_vae_model.discriminator(interpolated, training=True)

        gradients = tape.gradient(pred, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))

        return penalty

    @tf.function
    def train_discriminator_step(real_batch):
        """Train discriminator for one step."""
        batch_size = tf.shape(real_batch)[0]

        with tf.GradientTape() as tape:
            # Encode real data
            z_mean, z_log_var = ctgan_vae_model.encoder(real_batch, training=True)
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

            # Generate fake data
            noise = tf.random.normal((batch_size, ctgan_vae_model.noise_dim))
            fake_data = ctgan_vae_model.generator([z, noise], training=True)

            # Discriminator predictions
            real_pred = ctgan_vae_model.discriminator(real_batch, training=True)
            fake_pred = ctgan_vae_model.discriminator(fake_data, training=True)

            # Wasserstein loss
            w_dist = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

            # Gradient penalty
            gp = gradient_penalty(real_batch, fake_data)

            # Total discriminator loss
            d_loss = w_dist + gradient_penalty_weight * gp

        # Update discriminator
        gradients = tape.gradient(d_loss, ctgan_vae_model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(
            zip(gradients, ctgan_vae_model.discriminator.trainable_variables)
        )

        return d_loss, w_dist, gp

    @tf.function
    def train_generator_step(real_batch):
        """Train generator/encoder for one step."""
        batch_size = tf.shape(real_batch)[0]

        with tf.GradientTape() as tape:
            # Forward pass
            x_reconstructed, z_mean, z_log_var = ctgan_vae_model(real_batch, training=True)

            # VAE losses
            total_vae_loss, recon_loss, kl_loss = vae_loss(
                real_batch, x_reconstructed, z_mean, z_log_var, ctgan_vae_model.beta
            )

            # Adversarial loss
            fake_pred = ctgan_vae_model.discriminator(x_reconstructed, training=True)
            adv_loss = -tf.reduce_mean(fake_pred)

            # Combined generator loss
            g_loss = adv_loss + total_vae_loss

        # Update generator and encoder
        trainable_vars = (ctgan_vae_model.generator.trainable_variables +
                          ctgan_vae_model.encoder.trainable_variables)
        gradients = tape.gradient(g_loss, trainable_vars)
        gen_optimizer.apply_gradients(zip(gradients, trainable_vars))

        return g_loss, recon_loss, kl_loss

    # Training loop
    iterator = tqdm(range(epochs), desc='Training CTGAN-VAE') if verbose else range(epochs)

    for epoch in iterator:
        epoch_g_loss = []
        epoch_d_loss = []
        epoch_w_dist = []
        epoch_gp = []
        epoch_recon_loss = []
        epoch_kl_loss = []

        for batch in dataset:
            # Train discriminator n_critic times
            for _ in range(n_critic):
                d_loss, w_dist, gp = train_discriminator_step(batch)

            epoch_d_loss.append(d_loss.numpy())
            epoch_w_dist.append(w_dist.numpy())
            epoch_gp.append(gp.numpy())

            # Train generator once
            g_loss, recon_loss, kl_loss = train_generator_step(batch)

            epoch_g_loss.append(g_loss.numpy())
            epoch_recon_loss.append(recon_loss.numpy())
            epoch_kl_loss.append(kl_loss.numpy())

        # Record epoch metrics
        history['g_loss'].append(np.mean(epoch_g_loss))
        history['d_loss'].append(np.mean(epoch_d_loss))
        history['w_distance'].append(np.mean(epoch_w_dist))
        history['gp'].append(np.mean(epoch_gp))
        history['recon_loss'].append(np.mean(epoch_recon_loss))
        history['kl_loss'].append(np.mean(epoch_kl_loss))

        if verbose:
            iterator.set_postfix({
                'G_loss': f'{history["g_loss"][-1]:.4f}',
                'D_loss': f'{history["d_loss"][-1]:.4f}',
                'W_dist': f'{history["w_distance"][-1]:.4f}'
            })

    return history


def train_bigan(bigan_model,
                real_data: np.ndarray,
                epochs: int = 100,
                batch_size: int = 256,
                learning_rate: float = 2e-4,
                verbose: bool = True) -> Dict:
    """
    Train Bidirectional GAN.

    Parameters:
    -----------
    bigan_model : BiGAN
        BiGAN model to train
    real_data : array
        Real training data
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    verbose : bool
        Print training progress

    Returns:
    --------
    dict : Training history
    """
    gen_enc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.9)

    # Prepare dataset
    dataset = tf.data.Dataset.from_tensor_slices(real_data)
    dataset = dataset.shuffle(buffer_size=len(real_data))
    dataset = dataset.batch(batch_size)

    # Training history
    history = {
        'g_loss': [],
        'd_loss': []
    }

    @tf.function
    def train_discriminator_step(real_batch):
        """Train discriminator for one step."""
        batch_size = tf.shape(real_batch)[0]

        with tf.GradientTape() as tape:
            # Real pairs: (x_real, E(x_real))
            z_encoded = bigan_model.encoder(real_batch, training=True)

            # Fake pairs: (G(z), z)
            z_random = tf.random.normal((batch_size, bigan_model.latent_dim))
            x_generated = bigan_model.generator(z_random, training=True)

            # Discriminator predictions
            real_pair_pred = bigan_model.discriminator([real_batch, z_encoded], training=True)
            fake_pair_pred = bigan_model.discriminator([x_generated, z_random], training=True)

            # Binary cross-entropy loss
            real_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_pair_pred), real_pair_pred, from_logits=False
            )
            fake_loss = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_pair_pred), fake_pair_pred, from_logits=False
            )

            d_loss = tf.reduce_mean(real_loss + fake_loss)

        # Update discriminator
        gradients = tape.gradient(d_loss, bigan_model.discriminator.trainable_variables)
        disc_optimizer.apply_gradients(
            zip(gradients, bigan_model.discriminator.trainable_variables)
        )

        return d_loss

    @tf.function
    def train_generator_encoder_step(real_batch):
        """Train generator and encoder for one step."""
        batch_size = tf.shape(real_batch)[0]

        with tf.GradientTape() as tape:
            # Real pairs: (x_real, E(x_real))
            z_encoded = bigan_model.encoder(real_batch, training=True)

            # Fake pairs: (G(z), z)
            z_random = tf.random.normal((batch_size, bigan_model.latent_dim))
            x_generated = bigan_model.generator(z_random, training=True)

            # Discriminator predictions (fool discriminator)
            real_pair_pred = bigan_model.discriminator([real_batch, z_encoded], training=True)
            fake_pair_pred = bigan_model.discriminator([x_generated, z_random], training=True)

            # Reverse labels to fool discriminator
            real_loss = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(real_pair_pred), real_pair_pred, from_logits=False
            )
            fake_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_pair_pred), fake_pair_pred, from_logits=False
            )

            g_loss = tf.reduce_mean(real_loss + fake_loss)

        # Update generator and encoder
        trainable_vars = (bigan_model.generator.trainable_variables +
                          bigan_model.encoder.trainable_variables)
        gradients = tape.gradient(g_loss, trainable_vars)
        gen_enc_optimizer.apply_gradients(zip(gradients, trainable_vars))

        return g_loss

    # Training loop
    iterator = tqdm(range(epochs), desc='Training BiGAN') if verbose else range(epochs)

    for epoch in iterator:
        epoch_g_loss = []
        epoch_d_loss = []

        for batch in dataset:
            # Train discriminator
            d_loss = train_discriminator_step(batch)
            epoch_d_loss.append(d_loss.numpy())

            # Train generator and encoder
            g_loss = train_generator_encoder_step(batch)
            epoch_g_loss.append(g_loss.numpy())

        # Record epoch metrics
        history['g_loss'].append(np.mean(epoch_g_loss))
        history['d_loss'].append(np.mean(epoch_d_loss))

        if verbose:
            iterator.set_postfix({
                'G_loss': f'{history["g_loss"][-1]:.4f}',
                'D_loss': f'{history["d_loss"][-1]:.4f}'
            })

    return history


def generate_synthetic_data(model, n_samples: int) -> np.ndarray:
    """
    Generate synthetic data from any trained generative model.

    Parameters:
    -----------
    model : GenerativeModel
        Trained generative model (VAE, CTGAN-VAE, BiGAN)
    n_samples : int
        Number of synthetic samples to generate

    Returns:
    --------
    array : Synthetic data samples
    """
    return model.generate_samples(n_samples)


if __name__ == '__main__':
    print("Advanced training functions loaded successfully!")
    print("\nAvailable training functions:")
    print("  - train_vae()")
    print("  - train_ctgan_vae()")
    print("  - train_bigan()")
    print("  - generate_synthetic_data()")
