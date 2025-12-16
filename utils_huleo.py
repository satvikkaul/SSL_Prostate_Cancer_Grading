import numpy as np
try:
    from skimage import exposure
except ImportError:
    pass # scikit-image is optional if you don't use histogram matching

def get_random_patch_list(image_size, patch_size):
    """
    Generates a list of patch coordinates (y, x, size) for the input image size.
    Used for the Hide-and-Seek augmentation.
    """
    patches = []
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        h, w = image_size
        
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patches.append((y, x, patch_size))
    return patches

def random_hide(image, patch_list, hide_prob=0.5, mean=0):
    """
    Randomly hides patches in the image by setting them to the mean value.
    """
    img = image.copy()
    for (y, x, s) in patch_list:
        if np.random.rand() < hide_prob:
            img[y:y+s, x:x+s, :] = mean
    return img

def image_histogram_equalization(image, number_bins=256):
    """
    Applies histogram equalization to the image.
    """
    # Check if skimage is installed
    try:
        img_eq = exposure.equalize_hist(image, nbins=number_bins)
        return img_eq
    except NameError:
        print("Warning: scikit-image not installed, returning original image.")
        return image

def hist_match(source, template):
    """
    Matches the histogram of the source image to the template image.
    """
    try:
        matched = exposure.match_histograms(source, template, channel_axis=-1)
        return matched
    except NameError:
        print("Warning: scikit-image not installed, returning original image.")
        return source