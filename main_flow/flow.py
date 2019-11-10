import cv2
from feature_extraction.feature_extraction import extract_dense_features, extract_features
from preprocessing.preprocessing import *
from preprocessing.utils import *
from feature_extraction.gabor import gabor_features
from preprocessing.utils import show_image


def __process_features(img):
    """
    Process the resulting scales of the segmentation
    Parameters
    ----------
    img             numpy array containing the image
    all_scales      numpy array containing all the segmented ROIS in all scales

    Returns
    -------
    dataframe       dataframe of all the ROIs in the image

    """
    features = extract_features(img, hog=False)
    #features = extract_dense_features(img, 48)
    #features_textons = textons_features(img)
    features_gab = gabor_features(img)
    all_feat = np.concatenate((features_gab,features))
    return all_feat


def process_single_image(filename, debug=False):
    """
    Process a single image extracting the ROIS and the features
    Parameters
    ----------
    path            path where all the dataset is licated
    filename        file to extract the ROIS

    Returns
    -------
    all_scales      Segmentation ROIs of the image
    features        dataframe of all the features of the ROIs in the image
    img             numpy arrray containing the image

    """
    img = cv.imread(filename)
    #img_norm, _, _ = normalize_staining(img, debug=debug)
    features = __process_features(img)
    return features


def extract_ROI(roi_contour, img,padding = 0.05):
    """

    Parameters
    ----------
    roi_contour:    Contours points
        A contour of the images generated with the function findcontours
    img:            numpy array of 12 bits depth

    Returns
    -------
    roi:            Numpy array
        Image containing the required ROI of the image
    boundaries:     Numpy array
        Array containing the coordinates of the extracted ROI in the full image coordinates

    """
    # Get the boundaries of each region of interest
    x_b, y_b, w_b, h_b = cv2.boundingRect(roi_contour)

    padd_x = np.uint8(padding * w_b)
    padd_y = np.uint8(padding * h_b)

    x_b = max(x_b - padd_x, 0)
    y_b = max(y_b - padd_y, 0)

    # Adjust the boundaries so we get a surrounding region
    img_shape = img.shape
    x_m = min(x_b + w_b + 2 * padd_x, img_shape[1])
    y_m = min(y_b + h_b + 2 * padd_y, img_shape[0])

    # Extract the region of interest of the given contours
    roi = img[y_b:y_m, x_b:x_m]
    boundaries = x_b, y_b, w_b, h_b

    return roi, boundaries


def save_mask(roi, path, filename):
    """
    Save a mask of the ROI

    Parameters
    ----------
    roi         Mask of the lesion
    path        Path to save the file
    filename    Name of the file to save

    Returns     Nothing
    -------

    """
    directory = filename.split('/')[-3:]
    directory = [path, 'mask'] + directory
    seperator = '/'
    write_path = seperator.join(directory).replace('\\', '/')
    cv2.imwrite(write_path, roi*255)
    cv2.waitKey(1)
