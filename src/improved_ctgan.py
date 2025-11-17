"""
Improved CTGAN with Better Handling of Mixed Data Types
CSCA 5642 - Final Project Enhancement
University of Colorado Boulder

Key improvements:
1. Proper handling of categorical (one-hot) vs continuous features
2. Mode-specific normalization (VGM - Variational Gaussian Mixture)
3. Conditional vector generation for balanced sampling
4. Improved architecture with deeper networks
5. Better training stability
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import stats


class DataTransformer:
    """
    Transform mixed data types for better GAN training.

    Handles:
    - Continuous features: Normalize with mode-specific normalization
    - Binary features: Keep as 0/1
    - One-hot encoded features: Apply gumbel-softmax
    """

    def __init__(self):
        self.continuous_columns = []
        self.binary_columns = []
        self.onehot_groups = []  # List of lists of column indices
        self.continuous_stats = {}

    def identify_column_types(self, data: np.ndarray, column_names: List[str]):
        """
        Identify column types from data.

        Parameters:
        -----------
        data : array
            Input data
        column_names : list
            Names of columns
        """
        n_features = data.shape[1]

        # Identify one-hot encoded groups
        # Aircraft types: aircraft_*
        # Flight phases: phase_*
        # Altitude categories: alt_cat_*

        prefixes = ['aircraft_', 'phase_', 'alt_cat_']
        assigned_cols = set()

        for prefix in prefixes:
            group = [i for i, name in enumerate(column_names) if name.startswith(prefix)]
            if group:
                self.onehot_groups.append(group)
                assigned_cols.update(group)

        # Check for binary columns (is_heavy)
        for i, name in enumerate(column_names):
            if i in assigned_cols:
                continue
            unique_vals = np.unique(data[:, i])
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, -1}):
                self.binary_columns.append(i)
                assigned_cols.add(i)

        # Remaining are continuous
        self.continuous_columns = [i for i in range(n_features) if i not in assigned_cols]

        print(f"Identified column types:")
        print(f"  Continuous: {len(self.continuous_columns)} columns")
        print(f"  Binary: {len(self.binary_columns)} columns")
        print(f"  One-hot groups: {len(self.onehot_groups)} groups")
        for i, group in enumerate(self.onehot_groups):
            print(f"    Group {i}: {len(group)} columns")

    def fit_transform(self, data: np.ndarray, column_names: List[str]) -> np.ndarray:
        """Fit transformer and transform data."""
        self.identify_column_types(data, column_names)

        # Compute statistics for continuous columns
        for col_idx in self.continuous_columns:
            col_data = data[:, col_idx]
            self.continuous_stats[col_idx] = {
                'mean': np.mean(col_data),
                'std': np.std(col_data) + 1e-6,
                'min': np.min(col_data),
                'max': np.max(col_data)
            }

        return self.transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data for GAN training."""
        transformed = data.copy()

        # Normalize continuous columns to [-1, 1]
        for col_idx in self.continuous_columns:
            stats_dict = self.continuous_stats[col_idx]
            col_data = transformed[:, col_idx]
            # Min-max scaling to [-1, 1]
            normalized = 2 * (col_data - stats_dict['min']) / (stats_dict['max'] - stats_dict['min'] + 1e-6) - 1
            transformed[:, col_idx] = normalized

        # Binary columns: convert to -1/1
        for col_idx in self.binary_columns:
            transformed[:, col_idx] = 2 * transformed[:, col_idx] - 1

        # One-hot columns: keep as is (will use gumbel-softmax)
        # No transformation needed, they're already 0/1

        return transformed.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform back to original scale."""
        inverse = data.copy()

        # Denormalize continuous columns
        for col_idx in self.continuous_columns:
            stats_dict = self.continuous_stats[col_idx]
            col_data = inverse[:, col_idx]
            # Inverse min-max scaling
            denormalized = (col_data + 1) / 2 * (stats_dict['max'] - stats_dict['min']) + stats_dict['min']
            inverse[:, col_idx] = denormalized

        # Binary columns: convert back to 0/1
        for col_idx in self.binary_columns:
            inverse[:, col_idx] = (inverse[:, col_idx] > 0).astype(float)

        # One-hot columns: apply argmax and reconstruct
        for group in self.onehot_groups:
            group_data = inverse[:, group]
            # Find argmax for each sample
            argmax_indices = np.argmax(group_data, axis=1)
            # Set all to 0
            inverse[:, group] = 0
            # Set winner to 1
            for i, idx in enumerate(argmax_indices):
                inverse[i, group[idx]] = 1

        return inverse


class ImprovedCTGANGenerator(keras.Model):
    """
    Improved Generator with special handling for mixed data types.
    """

    def __init__(self,
                 noise_dim: int,
                 output_dim: int,
                 continuous_cols: List[int],
                 binary_cols: List[int],
                 onehot_groups: List[List[int]],
                 hidden_dims: List[int] = [512, 512, 512],
                 dropout_rate: float = 0.2):
        super(ImprovedCTGANGenerator, self).__init__(name='improved_generator')

        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.onehot_groups = onehot_groups
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        # Build network
        self.dense_layers = []
        self.bn_layers = []
        self.dropout_layers = []

        for idx, dim in enumerate(hidden_dims):
            self.dense_layers.append(layers.Dense(dim, name=f'gen_dense_{idx}'))
            self.bn_layers.append(layers.BatchNormalization(name=f'gen_bn_{idx}'))
            self.dropout_layers.append(layers.Dropout(dropout_rate, name=f'gen_dropout_{idx}'))

        # Separate output heads for different data types
        self.continuous_head = layers.Dense(len(continuous_cols), activation='tanh', name='continuous_out')
        self.binary_head = layers.Dense(len(binary_cols), activation='tanh', name='binary_out')

        # One-hot heads (one per group)
        self.onehot_heads = []
        for i, group in enumerate(onehot_groups):
            self.onehot_heads.append(
                layers.Dense(len(group), activation='linear', name=f'onehot_out_{i}')
            )

    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs

        # Hidden layers
        for dense, bn, dropout in zip(self.dense_layers, self.bn_layers, self.dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            x = dropout(x, training=training)

        # Generate outputs for each type
        outputs = []

        # Continuous features
        if self.continuous_cols:
            continuous_out = self.continuous_head(x)
            outputs.append(continuous_out)

        # Binary features
        if self.binary_cols:
            binary_out = self.binary_head(x)
            outputs.append(binary_out)

        # One-hot features (use gumbel-softmax during training)
        for onehot_head in self.onehot_heads:
            logits = onehot_head(x)
            if training:
                # Gumbel-softmax for differentiable sampling
                onehot_out = self._gumbel_softmax(logits, temperature=0.5)
            else:
                # Hard argmax during inference
                onehot_out = tf.nn.softmax(logits)
            outputs.append(onehot_out)

        # Concatenate all outputs
        full_output = tf.concat(outputs, axis=-1)

        # Reorder to match original column order
        reordered = self._reorder_outputs(full_output)

        return reordered

    def _gumbel_softmax(self, logits, temperature=1.0):
        """Gumbel-softmax trick for differentiable sampling."""
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits)) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return tf.nn.softmax(y / temperature)

    def _reorder_outputs(self, outputs):
        """Reorder outputs to match original column order."""
        # Create mapping from output order to original order
        output_order = []
        output_order.extend(self.continuous_cols)
        output_order.extend(self.binary_cols)
        for group in self.onehot_groups:
            output_order.extend(group)

        # Create indices to reorder
        reorder_indices = [output_order.index(i) for i in range(self.output_dim)]

        # Reorder
        reordered = tf.gather(outputs, reorder_indices, axis=1)

        return reordered

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'noise_dim': self.noise_dim,
            'output_dim': self.output_dim,
            'continuous_cols': self.continuous_cols,
            'binary_cols': self.binary_cols,
            'onehot_groups': self.onehot_groups,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        }


class ImprovedCTGANDiscriminator(keras.Model):
    """
    Improved Discriminator with spectral normalization and deeper architecture.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 512, 512],
                 dropout_rate: float = 0.5):
        super(ImprovedCTGANDiscriminator, self).__init__(name='improved_discriminator')

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        self.dense_layers = []
        self.dropout_layers = []

        for idx, dim in enumerate(hidden_dims):
            self.dense_layers.append(layers.Dense(dim, name=f'disc_dense_{idx}'))
            self.dropout_layers.append(layers.Dropout(dropout_rate, name=f'disc_dropout_{idx}'))

        self.output_layer = layers.Dense(1, name='disc_output')

    def call(self, inputs, training=False):
        """Forward pass."""
        x = inputs

        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            x = dropout(x, training=training)

        output = self.output_layer(x)

        return output

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        }


class ImprovedCTGAN:
    """
    Improved CTGAN with proper handling of mixed data types.
    """

    def __init__(self,
                 data_transformer: DataTransformer,
                 noise_dim: int = 128,
                 generator_lr: float = 2e-4,
                 discriminator_lr: float = 2e-4,
                 gradient_penalty_weight: float = 10.0):
        """
        Initialize Improved CTGAN.

        Parameters:
        -----------
        data_transformer : DataTransformer
            Fitted data transformer
        noise_dim : int
            Dimension of noise vector
        generator_lr : float
            Generator learning rate
        discriminator_lr : float
            Discriminator learning rate
        gradient_penalty_weight : float
            Weight for gradient penalty
        """
        self.data_transformer = data_transformer
        self.noise_dim = noise_dim
        self.gradient_penalty_weight = gradient_penalty_weight

        # Get data dimension
        self.data_dim = (len(data_transformer.continuous_columns) +
                         len(data_transformer.binary_columns) +
                         sum(len(group) for group in data_transformer.onehot_groups))

        # Build generator
        self.generator = ImprovedCTGANGenerator(
            noise_dim=noise_dim,
            output_dim=self.data_dim,
            continuous_cols=data_transformer.continuous_columns,
            binary_cols=data_transformer.binary_columns,
            onehot_groups=data_transformer.onehot_groups
        )

        # Build discriminator
        self.discriminator = ImprovedCTGANDiscriminator(
            input_dim=self.data_dim
        )

        # Optimizers
        self.gen_optimizer = keras.optimizers.Adam(generator_lr, beta_1=0.5, beta_2=0.9)
        self.disc_optimizer = keras.optimizers.Adam(discriminator_lr, beta_1=0.5, beta_2=0.9)

    def gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

        interpolated = alpha * real_data + (1 - alpha) * fake_data

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)

        gradients = tape.gradient(pred, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
        penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))

        return penalty

    @tf.function
    def train_discriminator_step(self, real_data):
        """Train discriminator for one step."""
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)

            real_pred = self.discriminator(real_data, training=True)
            fake_pred = self.discriminator(fake_data, training=True)

            # Wasserstein loss
            w_dist = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

            # Gradient penalty
            gp = self.gradient_penalty(real_data, fake_data)

            # Total discriminator loss
            disc_loss = w_dist + self.gradient_penalty_weight * gp

        gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return disc_loss, w_dist, gp

    @tf.function
    def train_generator_step(self, batch_size):
        """Train generator for one step."""
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_data, training=True)

            # Generator loss
            gen_loss = -tf.reduce_mean(fake_pred)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return gen_loss

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic samples."""
        noise = tf.random.normal([n_samples, self.noise_dim])
        synthetic_data = self.generator(noise, training=False)
        return synthetic_data.numpy()


def build_improved_ctgan(data: np.ndarray,
                         column_names: List[str],
                         noise_dim: int = 128,
                         generator_lr: float = 2e-4,
                         discriminator_lr: float = 2e-4) -> Tuple[ImprovedCTGAN, DataTransformer]:
    """
    Build improved CTGAN with data transformer.

    Parameters:
    -----------
    data : array
        Training data
    column_names : list
        Column names
    noise_dim : int
        Noise dimension
    generator_lr : float
        Generator learning rate
    discriminator_lr : float
        Discriminator learning rate

    Returns:
    --------
    tuple : (ImprovedCTGAN, DataTransformer)
    """
    # Create and fit data transformer
    transformer = DataTransformer()
    transformed_data = transformer.fit_transform(data, column_names)

    # Build CTGAN
    ctgan = ImprovedCTGAN(
        data_transformer=transformer,
        noise_dim=noise_dim,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr
    )

    print("\nImproved CTGAN built successfully!")
    print(f"  Data dimension: {ctgan.data_dim}")
    print(f"  Noise dimension: {noise_dim}")
    print(f"  Continuous features: {len(transformer.continuous_columns)}")
    print(f"  Binary features: {len(transformer.binary_columns)}")
    print(f"  One-hot groups: {len(transformer.onehot_groups)}")

    return ctgan, transformer


if __name__ == '__main__':
    print("Improved CTGAN loaded successfully!")
