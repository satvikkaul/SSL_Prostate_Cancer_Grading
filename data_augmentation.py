import cv2
import numpy as np

def translateit_fast_2d(image, offset, fill_value=0):
    """
    Translates the image by offset.
    offset: [dy, dx] (vertical, horizontal) - matching original logic where offset[0] was rows
    """
    rows, cols = image.shape[:2]
    # cv2.warpAffine expects [dx, dy] translation
    # Original code: offset[0] is rows (y), offset[1] is cols (x)
    dx = offset[1]
    dy = offset[0]
    
    M = np.float32([[1, 0, dx], [0, 1, dy]]) # type: ignore
    return cv2.warpAffine(image, M, (cols, rows), borderValue=fill_value) # type: ignore

def scaleit_2d(image, factor, isseg=False):
    """
    Scales the image.
    factor > 1.0: Zoom In (Center Crop)
    factor < 1.0: Zoom Out (Pad)
    """
    if factor == 1.0:
        return image

    height, width = image.shape[:2]
    interpolation = cv2.INTER_NEAREST if isseg else cv2.INTER_LINEAR

    if factor > 1.0:
        # Zoom in
        new_height, new_width = int(height * factor), int(width * factor)
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Crop Center
        start_h = (new_height - height) // 2
        start_w = (new_width - width) // 2
        
        # Handle potential rounding errors causing slice issues
        end_h = start_h + height
        end_w = start_w + width
        
        # Ensure we don't go out of bounds if integers round weirdly
        if end_h > new_height: start_h -= 1; end_h -= 1
        if end_w > new_width: start_w -= 1; end_w -= 1
            
        return resized[start_h:end_h, start_w:end_w]
        
    else:
        # Zoom out
        new_height, new_width = int(height * factor), int(width * factor)
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        # Pad
        if len(image.shape) == 3:
            canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((height, width), dtype=image.dtype)
            
        start_h = (height - new_height) // 2
        start_w = (width - new_width) // 2
        
        canvas[start_h:start_h+new_height, start_w:start_w+new_width] = resized
        return canvas

def resampleit(image, dims, isseg=False):
    """
    Resizes image to specific dims.
    """
    # dims is likely (height, width) or similar depending on usage.
    # Original used zoom with dims/shape factor.
    # Assuming dims is target shape.
    interpolation = cv2.INTER_NEAREST if isseg else cv2.INTER_LINEAR
    # cv2.resize takes (width, height)
    return cv2.resize(image, (dims[1], dims[0]), interpolation=interpolation)

def rotateit_2d(image, theta1, isseg=False):
    """
    Rotates image by theta1 degrees.
    """
    if theta1 == 0.0:
        return image
        
    rows, cols = image.shape[:2]
    # Rotation matrix around center
    M = cv2.getRotationMatrix2D((cols/2, rows/2), theta1, 1)
    interpolation = cv2.INTER_NEAREST if isseg else cv2.INTER_LINEAR
    return cv2.warpAffine(image, M, (cols, rows), flags=interpolation)

def intensifyit_2d(image, factor):
    """
    Multiplies intensity.
    """
    return image * float(factor)
