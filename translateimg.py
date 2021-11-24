import numpy as np

from pathlib import Path
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
import math 

inputPath = Path(r"C:\Users\wojciechostrowski\Desktop\Semestr7\Inzynierka\Aplikacja\NumberPlateRecognition\simpleimages\rotate")
inputFiles = inputPath.glob("**/*.jpg")

def rotate_and_crop(image, output_height, output_width, rotation_degree, do_crop):
    """Rotate the given image with the given rotation degree and crop for the black edges if necessary
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        rotation_degree: The degree of rotation on the image.
        do_crop: Do cropping if it is True.
    Returns:
        A rotated image.
    """
  
    # Rotate the given image with the given rotation degree
    if rotation_degree != 0:
        image = tfa.image.rotate(image, math.radians(rotation_degree), interpolation='BILINEAR')
      
        # Center crop to ommit black noise on the edges
        if do_crop == True:
            lrr_width, lrr_height = largest_rotated_rect(output_height, output_width, math.radians(rotation_degree))
            resized_image = tf.image.central_crop(image, float(lrr_height)/output_height)    
            image = tf.image.resize(resized_image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)
    
    return image

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )
    
for f in inputFiles:
    outputFile = inputPath / Path(f.stem + ".jpg")
    im = Image.open(f)
    width, height = im.size
    rotatedIm = rotate_and_crop(im, height, width, -10, True)
    print(rotatedIm.shape)
    tf.keras.preprocessing.image.save_img(outputFile,rotatedIm)
