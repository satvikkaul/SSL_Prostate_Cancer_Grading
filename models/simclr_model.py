"""
SimCLR Model Architecture

Implements the SimCLR framework with:
1. Base encoder (reuses existing architecture from variational_autoencoder.py)
2. Projection head (MLP for contrastive learning)
3. NT-Xent (InfoNCE) loss function
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class ProjectionHead(tf.keras.Model):
    """
    MLP projection head for SimCLR.
    
    Architecture: features → Dense(256, ReLU) → Dense(128)
    
    The projection head is discarded after pretraining;
    only the encoder is used for downstream tasks.
    """
    
    def __init__(self, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        
        self.dense1 = Dense(hidden_dim, use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        self.dense2 = Dense(output_dim, use_bias=False)
    
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class SimCLRModel(tf.keras.Model):
    """
    Complete SimCLR model: Encoder + Projection Head
    """
    
    def __init__(self, encoder, projection_hidden_dim=256, projection_output_dim=128):
        """
        Args:
            encoder: Base encoder network (from models.cae_model)
            projection_hidden_dim: Hidden dimension of projection head
            projection_output_dim: Output dimension of projection head (embedding size)
        """
        super(SimCLRModel, self).__init__()
        
        self.encoder = encoder
        self.projection_head = ProjectionHead(projection_hidden_dim, projection_output_dim)
    
    def call(self, x, training=False):
        """
        Forward pass through encoder and projection head.
        
        Args:
            x: Input image batch (B, H, W, 3)
            training: Whether in training mode
        
        Returns:
            Projection embeddings (B, projection_output_dim)
        """
        # Extract features using encoder
        features = self.encoder(x, training=training)  # (B, feature_dim)
        
        # Project to embedding space
        embeddings = self.projection_head(features, training=training)  # (B, 128)
        
        return embeddings


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    Also known as InfoNCE loss.
    
    This is the core loss function for SimCLR contrastive learning.
    
    Args:
        z_i: Embeddings from view 1 (batch_size, embedding_dim)
        z_j: Embeddings from view 2 (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling (default 0.5)
    
    Returns:
        Scalar loss value
    
    How it works:
        1. For each image pair (i, i'), compute similarity with all other images
        2. Positive pair: (i, i') should have high similarity
        3. Negative pairs: (i, j) where j != i should have low similarity
        4. Loss encourages positive pairs to be close, negatives to be far
    """
    batch_size = tf.shape(z_i)[0]
    
    # L2 normalize embeddings (projects to unit hypersphere)
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    
    # Concatenate embeddings from both views: [z_i; z_j]
    # Shape: (2*batch_size, embedding_dim)
    z = tf.concat([z_i, z_j], axis=0)
    
    # Compute similarity matrix: (2N, 2N)
    # similarity[i,j] = cosine_similarity(z[i], z[j])
    similarity_matrix = tf.matmul(z, z, transpose_b=True)
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Create labels for positive pairs
    # For batch of N:
    #   - Image i pairs with image (i + N)
    #   - Image (i + N) pairs with image i
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)
    
    # Remove self-similarity (diagonal elements)
    # We don't want to compare an image with itself
    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    similarity_matrix = tf.where(
        mask,
        tf.ones_like(similarity_matrix) * -1e9,  # Set diagonal to very negative
        similarity_matrix
    )
    
    # Compute cross-entropy loss
    # This treats the similarity scores as logits for a classification task
    # where the goal is to identify the positive pair
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=similarity_matrix
    )
    
    return tf.reduce_mean(loss)


def cosine_similarity_loss(z_i, z_j, temperature=0.5):
    """
    Alternative: Simple cosine similarity loss (easier to understand).
    Not used in standard SimCLR but can be useful for debugging.
    """
    # Normalize
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    
    # Compute cosine similarity
    similarity = tf.reduce_sum(z_i * z_j, axis=1)
    
    # We want similarity to be high (close to 1)
    # So minimize negative similarity
    loss = -tf.reduce_mean(similarity)
    
    return loss


class SimCLRTrainer:
    """
    Training wrapper for SimCLR with convenient methods.
    """
    
    def __init__(self, model, optimizer, temperature=0.5):
        self.model = model
        self.optimizer = optimizer
        self.temperature = temperature
        
        # Metrics
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    
    @tf.function
    def train_step(self, x_i, x_j):
        """
        Single training step.
        
        Args:
            x_i: First augmented view (batch_size, H, W, 3)
            x_j: Second augmented view (batch_size, H, W, 3)
        
        Returns:
            Loss value
        """
        with tf.GradientTape() as tape:
            # Forward pass
            z_i = self.model(x_i, training=True)
            z_j = self.model(x_j, training=True)
            
            # Compute contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=self.temperature)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        
        return loss
    
    @tf.function
    def val_step(self, x_i, x_j):
        """Validation step (no gradient computation)"""
        z_i = self.model(x_i, training=False)
        z_j = self.model(x_j, training=False)
        loss = nt_xent_loss(z_i, z_j, temperature=self.temperature)
        self.val_loss_metric.update_state(loss)
        return loss
    
    def reset_metrics(self):
        """Reset metrics at the start of each epoch"""
        self.train_loss_metric.reset_states()
        self.val_loss_metric.reset_states()


if __name__ == "__main__":
    print("Testing SimCLR model components...")
    
    # Test projection head
    projection = ProjectionHead(hidden_dim=256, output_dim=128)
    test_features = tf.random.normal((16, 256))  # (batch, features)
    embeddings = projection(test_features, training=True)
    print(f"✓ Projection head: {test_features.shape} → {embeddings.shape}")
    
    # Test NT-Xent loss
    z_i = tf.random.normal((16, 128))
    z_j = tf.random.normal((16, 128))
    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    print(f"✓ NT-Xent loss computed: {loss.numpy():.4f}")
    
    # Test with normalized embeddings (should give lower loss for similar embeddings)
    z_i_norm = tf.math.l2_normalize(z_i, axis=1)
    z_j_similar = z_i_norm + tf.random.normal((16, 128)) * 0.1  # Similar to z_i
    loss_similar = nt_xent_loss(z_i_norm, z_j_similar, temperature=0.5)
    print(f"✓ Loss with similar embeddings: {loss_similar.numpy():.4f}")
    
    print("\nSimCLR model components ready! ✓")
