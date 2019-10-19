import numpy as np
import cv2 as cv


def bring_to_256_levels(the_image):
    """
    Function normalize image to 0-255 range (8 bits)

    Parameters
    ----------
    the_image : numpy array Original image.

    Returns
    -------
    img_new : numpy array
        Normalized image
    """
    if the_image.max() == the_image.min():
        return the_image.astype(np.uint8)
    img_as_double = the_image.astype(float)
    normalized = np.divide((img_as_double - np.amin(img_as_double)), (np.amax(img_as_double) - np.amin(img_as_double)))
    normalized = normalized*(pow(2, 8) - 1)
    return normalized.astype(np.uint8)


def show_image(img_to_show, img_name, factor=1.0):
    """Function to display an image with a specific window name and specific resize factor

    Parameters
    ----------
    img_to_show : numpy array
        Image to be displayed
    img_name : string
        Name for the window
    factor : float
        Resize factor (between 0 and 1)

    """
    img_to_show = cv.resize(img_to_show, None, fx=factor, fy=factor)
    cv.imshow(img_name, img_to_show)