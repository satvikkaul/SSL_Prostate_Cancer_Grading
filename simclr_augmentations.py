"""
SimCLR Augmentation Pipeline for Histopathology Images

Strong augmentations designed for contrastive learning on medical images.
Based on Ciga et al. (2020) and Stacke et al. (2021) recommendations.
"""

import tensorflow as tf
import numpy as np
import cv2


class SimCLRAugmentation:
    """
    Strong augmentation pipeline for SimCLR contrastive learning.
    
    Key differences from standard augmentation:
    - Stronger color jittering (critical for histopathology stain variation)
    - Gaussian blur
    - Random cropping with resize
    - Combined geometric transforms
    """
    
    def __init__(self, img_size=128, crop_ratio_range=(0.8, 1.0)):
        self.img_size = img_size
        self.crop_ratio_range = crop_ratio_range
    
    def color_jitter(self, image, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        """
        Apply color jittering for stain normalization invariance.
        Critical for histopathology due to staining variations.
        """
        # Random brightness
        if np.random.rand() < 0.8:
            delta = np.random.uniform(-brightness, brightness)
            image = tf.image.adjust_brightness(image, delta)
        
        # Random contrast
        if np.random.rand() < 0.8:
            factor = np.random.uniform(1.0 - contrast, 1.0 + contrast)
            image = tf.image.adjust_contrast(image, factor)
        
        # Random saturation
        if np.random.rand() < 0.8:
            factor = np.random.uniform(1.0 - saturation, 1.0 + saturation)
            image = tf.image.adjust_saturation(image, factor)
        
        # Random hue
        if np.random.rand() < 0.8:
            delta = np.random.uniform(-hue, hue)
            image = tf.image.adjust_hue(image, delta)
        
        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    
    def random_crop_and_resize(self, image):
        """
        Random crop with resize back to original size.
        Helps model learn scale-invariant features.
        """
        crop_ratio = np.random.uniform(*self.crop_ratio_range)
        crop_size = int(self.img_size * crop_ratio)
        
        # Random crop
        image = tf.image.random_crop(
            image, 
            size=[crop_size, crop_size, 3]
        )
        
        # Resize back to original
        image = tf.image.resize(image, [self.img_size, self.img_size])
        return image
    
    def gaussian_blur(self, image, kernel_size=3, sigma_range=(0.1, 2.0)):
        """
        Apply Gaussian blur with random sigma.
        Used in 50% of augmentations per SimCLR paper.
        """
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(*sigma_range)
            # Simple box blur approximation (faster than Gaussian)
            image = tf.nn.avg_pool2d(
                tf.expand_dims(image, 0),
                ksize=kernel_size,
                strides=1,
                padding='SAME'
            )
            image = tf.squeeze(image, 0)
        return image
    
    def geometric_transforms(self, image):
        """
        Apply random geometric transformations.
        """
        # Random horizontal flip
        if np.random.rand() < 0.5:
            image = tf.image.flip_left_right(image)
        
        # Random vertical flip (common in histopathology)
        if np.random.rand() < 0.5:
            image = tf.image.flip_up_down(image)
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            image = tf.image.rot90(image, k=k)
        
        return image
    
    def __call__(self, image):
        """
        Apply full augmentation pipeline.
        
        Args:
            image: Input image tensor (H, W, 3), normalized to [0, 1]
        
        Returns:
            Augmented image tensor
        """
        # Ensure image is float32
        image = tf.cast(image, tf.float32)
        
        # Apply augmentations in sequence
        image = self.random_crop_and_resize(image)
        image = self.color_jitter(image)
        image = self.geometric_transforms(image)
        image = self.gaussian_blur(image)
        
        return image


def create_simclr_augmentation_pair(image, augmenter):
    """
    Create two different augmented views of the same image.
    This is the core of contrastive learning.
    
    Args:
        image: Input image (H, W, 3), normalized to [0, 1]
        augmenter: SimCLRAugmentation instance
    
    Returns:
        view1, view2: Two differently augmented versions
    """
    view1 = augmenter(image)
    view2 = augmenter(image)
    return view1, view2


# Numpy-compatible version for data generator
class SimCLRAugmentationNumpy:
    """
    Numpy-based augmentation for compatibility with existing data generator.
    """
    
    def __init__(self, img_size=128):
        self.img_size = img_size
    
    def color_jitter_np(self, image):
        """Color jittering using numpy/cv2"""
        img = image.copy()
        
        # Brightness
        if np.random.rand() < 0.8:
            factor = np.random.uniform(0.6, 1.4)
            img = np.clip(img * factor, 0, 1)
        
        # Contrast (via histogram stretching)
        if np.random.rand() < 0.8:
            factor = np.random.uniform(0.6, 1.4)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 1)
        
        return img
    
    def random_crop_and_resize_np(self, image):
        """Random crop with resize using cv2"""
        h, w = image.shape[:2]
        crop_ratio = np.random.uniform(0.8, 1.0)
        crop_size = int(min(h, w) * crop_ratio)
        
        # Random crop position
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        
        cropped = image[top:top+crop_size, left:left+crop_size]
        resized = cv2.resize(cropped, (self.img_size, self.img_size))
        
        return resized
    
    def geometric_transforms_np(self, image):
        """Geometric transforms using numpy"""
        # Horizontal flip
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
        
        # Vertical flip
        if np.random.rand() < 0.5:
            image = np.flipud(image)
        
        # Rotation (90 degree increments)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k)
        
        return image
    
    def __call__(self, image):
        """Apply augmentation pipeline"""
        image = self.random_crop_and_resize_np(image)
        image = self.color_jitter_np(image)
        image = self.geometric_transforms_np(image)
        return image


if __name__ == "__main__":
    # Test augmentation
    print("Testing SimCLR augmentation pipeline...")
    
    # Create dummy image
    test_image = np.random.rand(128, 128, 3).astype(np.float32)
    
    # TensorFlow version
    augmenter_tf = SimCLRAugmentation(img_size=128)
    view1, view2 = create_simclr_augmentation_pair(test_image, augmenter_tf)
    print(f"✓ TensorFlow augmentation: {view1.shape}, {view2.shape}")
    
    # Numpy version
    augmenter_np = SimCLRAugmentationNumpy(img_size=128)
    aug_image = augmenter_np(test_image)
    print(f"✓ Numpy augmentation: {aug_image.shape}")
    
    print("\nAugmentation pipeline ready! ✓")
