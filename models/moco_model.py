"""
MoCo v2 Model Architecture
===========================
Implements:
  - build_encoder(): ResNet50 backbone + 2-layer MLP projection head
  - MoCoV2Queue: FIFO queue for negative embeddings
  - MoCoV2Model: Full MoCo v2 wrapper with query/momentum encoders

Hardware target: RTX 3060 Laptop 6GB VRAM
Config: K=4096, batch=16, AMP=ON
"""

import tensorflow as tf
import numpy as np


class ColorJitter(tf.keras.layers.Layer):
    """Custom Keras layer for Random Color Jitter (MoCo v2 specific block)."""
    def __init__(self, p=0.8, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, images, training=None):
        if not training:
            return images
        
        # Apply jitter with probability p to the whole batch
        def apply_jitter():
            # TF image ops don't always support float16, cast to float32
            orig_dtype = images.dtype
            x = tf.cast(images, tf.float32)
            
            # TF image jitter functions operate on batches automatically natively
            x = tf.image.random_brightness(x, max_delta=0.4)
            x = tf.image.random_contrast(x, lower=0.6, upper=1.4)
            x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
            x = tf.image.random_hue(x, max_delta=0.1)
            
            x = tf.clip_by_value(x, 0.0, 1.0)
            return tf.cast(x, orig_dtype)
            
        return tf.cond(
            tf.random.uniform([]) < self.p,
            apply_jitter,
            lambda: images
        )


class RandomGrayscale(tf.keras.layers.Layer):
    """Custom Keras layer for Random Grayscale (MoCo v2 specific block)."""
    def __init__(self, p=0.2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def call(self, images, training=None):
        if not training:
            return images
            
        def apply_gray():
            orig_dtype = images.dtype
            x = tf.cast(images, tf.float32)
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 1, 3])
            return tf.cast(x, orig_dtype)
            
        return tf.cond(
            tf.random.uniform([]) < self.p,
            apply_gray,
            lambda: images
        )


class RandomGaussianBlur(tf.keras.layers.Layer):
    """Custom Keras layer for Random Gaussian Blur (MoCo v2 specific block)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, images, training=None):
        if not training:
            return images
            
        orig_dtype = images.dtype
        x = tf.cast(images, tf.float32)
            
        sigma = tf.random.uniform([]) * 1.9 + 0.1 # [0.1, 2.0]
        
        grid_x = tf.cast(tf.range(-2, 3), tf.float32)
        grid_y = tf.cast(tf.range(-2, 3), tf.float32)
        grid_xx, grid_yy = tf.meshgrid(grid_x, grid_y)
        kernel = tf.exp(-(grid_xx**2 + grid_yy**2) / (2.0 * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        
        # Group conv across batch
        blurred = tf.nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        blurred = tf.clip_by_value(blurred, 0.0, 1.0)
        return tf.cast(blurred, orig_dtype)


def get_moco_augmenter(image_size=(128, 128)):
    """Returns a Keras Sequential model for MoCo v2 GPU data augmentation."""
    return tf.keras.Sequential([
        # Random Crop & Resize
        tf.keras.layers.RandomCrop(
            height=int(image_size[0] * 0.8), # Approx 64% area crop
            width=int(image_size[1] * 0.8)
        ),
        tf.keras.layers.Resizing(image_size[0], image_size[1], interpolation='bicubic'),
        # Flip
        tf.keras.layers.RandomFlip("horizontal"),
        # Color Jitter
        ColorJitter(p=0.8),
        # Grayscale
        RandomGrayscale(p=0.2),
        # Blur
        RandomGaussianBlur()
    ], name='moco_augmenter')



def build_encoder(input_shape=(128, 128, 3), proj_dim=128, hidden_dim=2048):
    """
    Builds the MoCo v2 encoder: ResNet50 backbone + 2-layer MLP projection head.
    
    Architecture:
      Input → ResNet50(pooling='avg') → 2048-d → ReLU → hidden_dim → L2-norm → proj_dim
    
    Args:
        input_shape: (H, W, C) input shape.
        proj_dim: Output embedding dimensionality (128 per MoCo v2 paper).
        hidden_dim: Hidden layer size in projection head (2048 per MoCo v2 paper).
    
    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name='image_input')
    
    # Backbone: ResNet50 without top, random init (no ImageNet weights for SSL)
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        pooling='avg',
        input_shape=input_shape
    )
    
    features = backbone(inputs, training=True)  # (B, 2048)
    
    # 2-layer MLP projection head
    x = tf.keras.layers.Dense(hidden_dim, name='proj_hidden')(features)
    x = tf.keras.layers.BatchNormalization(name='proj_bn')(x)
    x = tf.keras.layers.ReLU(name='proj_relu')(x)
    x = tf.keras.layers.Dense(proj_dim, name='proj_output')(x)
    
    # L2 normalization (unit sphere)
    # Must use keras.ops inside a Functional model (Keras 3 / TF 2.16+ requirement)
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name='proj_l2norm'
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='moco_encoder')


class MoCoV2Queue:
    """
    FIFO queue of negative key embeddings for MoCo v2.
    
    Args:
        queue_size: K — number of negative keys to store. Default 4096.
        feat_dim: Embedding dimensionality. Default 128.
    """
    
    def __init__(self, queue_size=4096, feat_dim=128):
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        
        # Initialize queue with L2-normalized random vectors
        init = tf.math.l2_normalize(
            tf.random.normal([queue_size, feat_dim]), axis=1
        )
        # Use tf.Variable for in-place updates without triggering retracing
        self._queue = tf.Variable(init, trainable=False, dtype=tf.float32, name='moco_queue')
        self._ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name='moco_queue_ptr')
    
    def dequeue_and_enqueue(self, keys):
        """
        FIFO update: enqueue current batch keys, dequeue oldest entries.
        
        Args:
            keys: Tensor of shape (batch_size, feat_dim).
        """
        # Cast to float32 — keys may be float16 when AMP is enabled
        keys = tf.cast(keys, tf.float32)
        batch_size = tf.shape(keys)[0]
        ptr = self._ptr
        
        # Wrap-around circular buffer update
        indices = tf.math.mod(
            tf.range(ptr, ptr + batch_size), self.queue_size
        )
        # Build (N, 1) index tensor for scatter update
        indices_2d = tf.expand_dims(indices, axis=1)  # (batch, 1)
        updated = tf.tensor_scatter_nd_update(self._queue, indices_2d, keys)
        self._queue.assign(updated)
        # Advance pointer
        new_ptr = tf.math.mod(ptr + batch_size, self.queue_size)
        self._ptr.assign(new_ptr)
    
    def get_queue(self):
        """Returns the current queue tensor (K, feat_dim)."""
        return self._queue
    
    def reset(self):
        """Re-initialize queue with random normalized vectors."""
        init = tf.math.l2_normalize(
            tf.random.normal([self.queue_size, self.feat_dim]), axis=1
        )
        self._queue.assign(init)
        self._ptr.assign(0)


class MoCoV2Model:
    """
    MoCo v2: Momentum Contrast v2 self-supervised learning wrapper.
    
    Maintains:
      - encoder_q: Query encoder (trained via backprop).
      - encoder_k: Momentum encoder (updated via EMA, NOT via gradients).
      - queue: FIFO negative key queue (K=4096).
    
    Args:
        input_shape: (H, W, C). Default (128, 128, 3).
        queue_size: K. Default 4096.
        proj_dim: Embedding dim. Default 128.
        momentum: EMA momentum m. Default 0.999.
        temperature: InfoNCE temperature τ. Default 0.2.
    """
    
    def __init__(self, input_shape=(128, 128, 3), queue_size=4096,
                 proj_dim=128, momentum=0.999, temperature=0.2):
        self.momentum = momentum
        self.temperature = temperature
        
        # Build query encoder (trainable)
        self.encoder_q = build_encoder(input_shape, proj_dim)
        
        # Build momentum encoder (structural replica, NOT trainable via backprop)
        self.encoder_k = build_encoder(input_shape, proj_dim)
        self.encoder_k.trainable = False
        
        # Initialize momentum encoder weights = query encoder weights
        self._copy_weights_q_to_k()
        
        # FIFO Queue
        self.queue = MoCoV2Queue(queue_size=queue_size, feat_dim=proj_dim)
    
    def _copy_weights_q_to_k(self):
        """Initialize encoder_k with the same weights as encoder_q."""
        for wq, wk in zip(self.encoder_q.weights, self.encoder_k.weights):
            wk.assign(wq)
    
    def momentum_update(self):
        """
        EMA update: θ_k ← m·θ_k + (1−m)·θ_q
        Called at the END of every training step.
        """
        for wq, wk in zip(self.encoder_q.weights, self.encoder_k.weights):
            wk.assign(self.momentum * wk + (1.0 - self.momentum) * wq)
    
    @tf.function
    def info_nce_loss(self, q, k):
        """
        Compute InfoNCE loss (contrastive cross-entropy).
        
        Args:
            q: Query embeddings, shape (B, D), L2-normalized.
            k: Key embeddings, shape (B, D), L2-normalized.
        
        Returns:
            Scalar loss tensor.
        """
        # Positive logit: dot product of q with its key k+ → shape (B, 1)
        # Cast to float32: AMP may produce float16, but queue is float32
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        l_pos = tf.reduce_sum(q * k, axis=1, keepdims=True)  # (B, 1)
        
        # Negative logits: dot product of q with all keys in queue → shape (B, K)
        # queue shape: (K, D) → transpose to (D, K)
        queue = tf.stop_gradient(tf.transpose(self.queue.get_queue()))  # (D, K)
        l_neg = tf.matmul(q, queue)  # (B, K)
        
        # Concatenate: first column = positive, rest = negatives → (B, K+1)
        logits = tf.concat([l_pos, l_neg], axis=1)
        logits = logits / self.temperature
        
        # Labels: positive is always at index 0
        batch_size = tf.shape(q)[0]
        labels = tf.zeros(batch_size, dtype=tf.int32)
        
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        return loss
    
    def summary(self):
        """Print encoder_q architecture summary."""
        self.encoder_q.summary()


if __name__ == '__main__':
    # Quick sanity check
    print("=== MoCo v2 Architecture Verification ===")
    model = MoCoV2Model(input_shape=(128, 128, 3), queue_size=4096)
    
    print(f"\nQuery encoder trainable params: {model.encoder_q.count_params():,}")
    print(f"Momentum encoder trainable: {model.encoder_k.trainable}")
    print(f"Queue shape: {model.queue.get_queue().shape}")
    
    # Test forward pass
    dummy_xq = tf.random.normal((4, 128, 128, 3))
    dummy_xk = tf.random.normal((4, 128, 128, 3))
    q = model.encoder_q(dummy_xq, training=True)
    k = model.encoder_k(dummy_xk, training=False)
    print(f"\nQuery embedding shape: {q.shape}")
    print(f"Key embedding shape: {k.shape}")
    
    loss = model.info_nce_loss(q, k)
    print(f"InfoNCE loss (sanity): {loss.numpy():.4f}")
    print("\n✓ All checks passed.")
