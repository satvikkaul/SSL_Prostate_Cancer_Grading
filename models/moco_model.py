"""
MoCo v2 model architecture and augmentation pipeline.

Implements:
  - build_encoder(): ResNet50 backbone + 2-layer MLP projection head
  - MoCoV2Queue: FIFO queue for negative embeddings
  - MoCoV2Model: Full MoCo v2 wrapper with query/momentum encoders

Hardware target: RTX 3060 Laptop 6GB VRAM
Config: K=4096, batch=16, AMP=ON
"""

import tensorflow as tf


def _batch_output_spec(images):
    return tf.TensorSpec(shape=images.shape[1:], dtype=images.dtype)


class RandomResizedCrop(tf.keras.layers.Layer):
    """Per-image random crop with resize back to the target resolution."""

    def __init__(self, image_size=(128, 128), scale=(0.8, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.scale = scale

    def _crop_single(self, image):
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        scale = tf.random.uniform([], minval=self.scale[0], maxval=self.scale[1])
        crop_height = tf.maximum(
            1, tf.cast(tf.round(tf.cast(height, tf.float32) * scale), tf.int32)
        )
        crop_width = tf.maximum(
            1, tf.cast(tf.round(tf.cast(width, tf.float32) * scale), tf.int32)
        )

        offset_height = tf.random.uniform([], 0, height - crop_height + 1, dtype=tf.int32)
        offset_width = tf.random.uniform([], 0, width - crop_width + 1, dtype=tf.int32)

        cropped = tf.image.crop_to_bounding_box(
            image,
            offset_height,
            offset_width,
            crop_height,
            crop_width,
        )
        resized = tf.image.resize(cropped, self.image_size, method="bicubic")
        return tf.cast(resized, image.dtype)

    def call(self, images, training=None):
        if not training:
            return images
        return tf.map_fn(self._crop_single, images, fn_output_signature=_batch_output_spec(images))


class ColorJitter(tf.keras.layers.Layer):
    """Per-image color jitter with stochastic application."""

    def __init__(self, p=0.8, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def _jitter_single(self, image):
        def apply_jitter():
            orig_dtype = image.dtype
            x = tf.cast(image, tf.float32)
            x = tf.image.random_brightness(x, max_delta=0.4)
            x = tf.image.random_contrast(x, lower=0.6, upper=1.4)
            x = tf.image.random_saturation(x, lower=0.6, upper=1.4)
            x = tf.image.random_hue(x, max_delta=0.1)
            x = tf.clip_by_value(x, 0.0, 1.0)
            return tf.cast(x, orig_dtype)

        return tf.cond(tf.random.uniform([]) < self.p, apply_jitter, lambda: image)

    def call(self, images, training=None):
        if not training:
            return images
        return tf.map_fn(self._jitter_single, images, fn_output_signature=_batch_output_spec(images))


class RandomGrayscale(tf.keras.layers.Layer):
    """Per-image grayscale conversion with stochastic application."""

    def __init__(self, p=0.2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def _gray_single(self, image):
        def apply_gray():
            orig_dtype = image.dtype
            x = tf.cast(image, tf.float32)
            x = tf.image.rgb_to_grayscale(x)
            x = tf.tile(x, [1, 1, 3])
            return tf.cast(x, orig_dtype)

        return tf.cond(tf.random.uniform([]) < self.p, apply_gray, lambda: image)

    def call(self, images, training=None):
        if not training:
            return images
        return tf.map_fn(self._gray_single, images, fn_output_signature=_batch_output_spec(images))


class RandomGaussianBlur(tf.keras.layers.Layer):
    """Per-image Gaussian blur with stochastic application."""

    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def _blur_single(self, image):
        def apply_blur():
            orig_dtype = image.dtype
            x = tf.cast(image, tf.float32)
            sigma = tf.random.uniform([]) * 1.9 + 0.1  # [0.1, 2.0]

            grid_x = tf.cast(tf.range(-2, 3), tf.float32)
            grid_y = tf.cast(tf.range(-2, 3), tf.float32)
            grid_xx, grid_yy = tf.meshgrid(grid_x, grid_y)
            kernel = tf.exp(-(grid_xx**2 + grid_yy**2) / (2.0 * sigma**2))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.expand_dims(tf.expand_dims(kernel, -1), -1)
            kernel = tf.tile(kernel, [1, 1, 3, 1])

            blurred = tf.nn.depthwise_conv2d(
                tf.expand_dims(x, axis=0),
                kernel,
                strides=[1, 1, 1, 1],
                padding="SAME",
            )[0]
            blurred = tf.clip_by_value(blurred, 0.0, 1.0)
            return tf.cast(blurred, orig_dtype)

        return tf.cond(tf.random.uniform([]) < self.p, apply_blur, lambda: image)

    def call(self, images, training=None):
        if not training:
            return images
        return tf.map_fn(self._blur_single, images, fn_output_signature=_batch_output_spec(images))


def get_moco_augmenter(image_size=(128, 128)):
    """Returns the MoCo v2 GPU augmentation pipeline."""

    return tf.keras.Sequential(
        [
            RandomResizedCrop(image_size=image_size, scale=(0.8, 1.0)),
            tf.keras.layers.RandomFlip("horizontal"),
            ColorJitter(p=0.8),
            RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=0.5),
        ],
        name="moco_augmenter",
    )


def build_encoder(input_shape=(128, 128, 3), proj_dim=128, hidden_dim=2048):
    """
    Build the MoCo v2 encoder: ResNet50 backbone + 2-layer MLP projection head.
    """

    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        pooling="avg",
        input_shape=input_shape,
    )

    # Let the caller control BatchNorm behavior through the model training flag.
    features = backbone(inputs)

    x = tf.keras.layers.Dense(hidden_dim, name="proj_hidden")(features)
    x = tf.keras.layers.BatchNormalization(name="proj_bn")(x)
    x = tf.keras.layers.ReLU(name="proj_relu")(x)
    x = tf.keras.layers.Dense(proj_dim, name="proj_output")(x)
    outputs = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="proj_l2norm",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="moco_encoder")


class MoCoV2Queue:
    """FIFO queue of negative key embeddings for MoCo v2."""

    def __init__(self, queue_size=4096, feat_dim=128):
        self.queue_size = queue_size
        self.feat_dim = feat_dim

        init = tf.math.l2_normalize(tf.random.normal([queue_size, feat_dim]), axis=1)
        self._queue = tf.Variable(init, trainable=False, dtype=tf.float32, name="moco_queue")
        self._ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name="moco_queue_ptr")

    def dequeue_and_enqueue(self, keys):
        keys = tf.cast(keys, tf.float32)
        batch_size = tf.shape(keys)[0]
        ptr = self._ptr

        indices = tf.math.mod(tf.range(ptr, ptr + batch_size), self.queue_size)
        indices_2d = tf.expand_dims(indices, axis=1)
        updated = tf.tensor_scatter_nd_update(self._queue, indices_2d, keys)
        self._queue.assign(updated)
        self._ptr.assign(tf.math.mod(ptr + batch_size, self.queue_size))

    def get_queue(self):
        return self._queue

    def get_ptr(self):
        return self._ptr

    def set_state(self, queue, ptr):
        self._queue.assign(tf.cast(queue, tf.float32))
        self._ptr.assign(tf.cast(ptr, tf.int32))

    def reset(self):
        init = tf.math.l2_normalize(tf.random.normal([self.queue_size, self.feat_dim]), axis=1)
        self._queue.assign(init)
        self._ptr.assign(0)


class MoCoV2Model:
    """MoCo v2 wrapper with query encoder, momentum encoder, and negative queue."""

    def __init__(
        self,
        input_shape=(128, 128, 3),
        queue_size=4096,
        proj_dim=128,
        momentum=0.999,
        temperature=0.2,
    ):
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_q = build_encoder(input_shape, proj_dim)
        self.encoder_k = build_encoder(input_shape, proj_dim)
        self.encoder_k.trainable = False

        self._copy_weights_q_to_k()
        self.queue = MoCoV2Queue(queue_size=queue_size, feat_dim=proj_dim)

    def _copy_weights_q_to_k(self):
        for wq, wk in zip(self.encoder_q.weights, self.encoder_k.weights):
            wk.assign(wq)

    def sync_key_encoder(self):
        self._copy_weights_q_to_k()

    def momentum_update(self):
        for wq, wk in zip(self.encoder_q.weights, self.encoder_k.weights):
            wk.assign(self.momentum * wk + (1.0 - self.momentum) * wq)

    def info_nce_loss(self, q, k):
        # Keep the loss eager-friendly. On some CPU-only Apple Silicon setups,
        # tracing this small function can stall for a long time while the eager
        # version runs immediately.
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        l_pos = tf.reduce_sum(q * k, axis=1, keepdims=True)

        queue = tf.stop_gradient(tf.transpose(self.queue.get_queue()))
        l_neg = tf.matmul(q, queue)

        logits = tf.concat([l_pos, l_neg], axis=1)
        logits = logits / self.temperature

        batch_size = tf.shape(q)[0]
        labels = tf.zeros(batch_size, dtype=tf.int32)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )

    def summary(self):
        self.encoder_q.summary()


if __name__ == "__main__":
    print("=== MoCo v2 Architecture Verification ===")
    model = MoCoV2Model(input_shape=(128, 128, 3), queue_size=4096)

    print(f"\nQuery encoder trainable params: {model.encoder_q.count_params():,}")
    print(f"Momentum encoder trainable: {model.encoder_k.trainable}")
    print(f"Queue shape: {model.queue.get_queue().shape}")

    dummy_xq = tf.random.normal((4, 128, 128, 3))
    dummy_xk = tf.random.normal((4, 128, 128, 3))
    q = model.encoder_q(dummy_xq, training=True)
    k = model.encoder_k(dummy_xk, training=False)
    print(f"\nQuery embedding shape: {q.shape}")
    print(f"Key embedding shape: {k.shape}")

    loss = model.info_nce_loss(q, k)
    print(f"InfoNCE loss (sanity): {loss.numpy():.4f}")
    print("\nAll checks passed.")
