import numpy as np
import cv2 as cv
import mahotas as mt
from radiomics import featureextractor
import SimpleITK as sitk
from math import copysign, log10
from skimage.feature import hog
from preprocessing.utils import bring_to_256_levels
from skimage.feature import daisy
import sys


def get_elongation(m):
    """
    Compute elongation from the moments of a shape
    (From: https://stackoverflow.com/questions/14854592/retrieve-elongation-feature-in-python-opencv-what-kind-of-moment-it-supposed-to)
    Parameters
    ----------
    m               Moments

    Returns
    -------
    -               Elongation value
    """

    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    return (x + y ** 0.5) / (x - y ** 0.5)


def get_geometrical_features(contour):
    """
    Extract geometrical features of a ROI

    Parameters
    ----------
    contour     Contour containing the ROI

    Returns
    -------
    geom_feat      all extracted geometrical features of a ROI
    """

    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    ellipse = cv.fitEllipse(contour)
    _, axes, _ = ellipse
    major_axis_length = max(axes)
    minor_axis_length = min(axes)

    compactness = (4 * np.pi * area) / (perimeter ** 2)
    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)

    # Discrete compactness (https://core.ac.uk/download/pdf/82756900.pdf)
    cd = (4 * area - perimeter) / 2
    cd_min = area - 1
    cd_max = (4 * area - 4 * np.sqrt(area)) / 2
    cd_n = (cd - cd_min) / (cd_max - cd_min)

    # Elongation
    m = cv.moments(contour)
    elongation = get_elongation(m)
    equi_diameter = np.sqrt(4 * area / np.pi)
    geom_feat = np.array([equi_diameter, compactness, elongation, eccentricity, cd_n])
    return geom_feat


def get_color_based_features(roi_color):
    """
    Calculates features in different sub colorspaces`
    Parameters
    ----------
    roi_color      color image

    Returns         mean and standar deviation on each color space
    -------

    """

    roi_color_double = roi_color.astype(float)
    # Compute color features according to Celebi2007
    all_color_spaces = np.zeros((roi_color.shape[0], roi_color.shape[1], 18),
                                dtype=np.float32)  # 18 because there are 6 color spaces with 3 channels each
    all_color_spaces[:, :, 0:3] = roi_color_double
    # Compute normalized RGB
    the_sum = roi_color_double[:, :, 2] + roi_color_double[:, :, 1] + roi_color_double[:, :, 0] + 0.00001
    all_color_spaces[:, :, 3] = roi_color_double[:, :, 2] / the_sum * 255.0
    all_color_spaces[:, :, 4] = roi_color_double[:, :, 1] / the_sum * 255.0
    all_color_spaces[:, :, 5] = roi_color_double[:, :, 0] / the_sum * 255.0
    # Compute HSV
    all_color_spaces[:, :, 6:9] = cv.cvtColor(roi_color, cv.COLOR_BGR2HSV).astype(np.float32)
    # Compute CIEluv
    all_color_spaces[:, :, 9:12] = cv.cvtColor(roi_color, cv.COLOR_BGR2Luv).astype(np.float32)
    # Compute Ohta (I1/2/3)
    all_color_spaces[:, :, 12] = (1 / 3) * roi_color_double[:, :, 2] + (1 / 3) * roi_color_double[:, :, 1] + (
                1 / 3) * roi_color_double[:, :, 0]
    all_color_spaces[:, :, 13] = (1 / 2) * roi_color_double[:, :, 2] + (1 / 2) * roi_color_double[:, :, 0]
    all_color_spaces[:, :, 14] = (-1) * (1 / 4) * roi_color_double[:, :, 2] + (1 / 2) * roi_color_double[:, :, 1] - (
                1 / 4) * roi_color_double[:, :, 0]
    # Compute l1/2/3
    denominator_l = (roi_color_double[:, :, 2] - roi_color_double[:, :, 1]) ** 2 + (
                roi_color_double[:, :, 2] - roi_color_double[:, :, 0]) ** 2 + (
                                roi_color_double[:, :, 1] - roi_color_double[:, :, 0]) ** 2 + 0.00001
    all_color_spaces[:, :, 15] = (roi_color_double[:, :, 2] - roi_color_double[:, :, 1]) ** 2 / (denominator_l)
    all_color_spaces[:, :, 16] = (roi_color_double[:, :, 2] - roi_color_double[:, :, 0]) ** 2 / (denominator_l)
    all_color_spaces[:, :, 17] = (roi_color_double[:, :, 2] - roi_color_double[:, :, 0]) ** 2 / (denominator_l)

    # Compute mean and std for each channel of each color space
    means_and_stds = np.zeros((1, 18 * 2))
    ii = 0
    for i in range(all_color_spaces.shape[2]):
        means_and_stds[0, ii] = np.mean(all_color_spaces[:, :, i])
        means_and_stds[0, ii + 1] = np.std(all_color_spaces[:, :, i])
        ii += 2

    return means_and_stds.reshape(means_and_stds.shape[1], )


def get_texture_features(roi_gray, mask):
    """
    Extract texture features from ROI

    Parameters
    ----------
    roi_gray            Region of interest of the image (gray scale)
    mask                Binary version of the ROI

    Returns
    -------
    texture_features    All extracted texture features of a ROI
    """
    # First, get GLRLM features
    data_spacing = [1, 1, 1]

    # Convert numpy arrays to sitk so that extractor.execute can be employed for GLRLM features
    sitk_img = sitk.GetImageFromArray(roi_gray)
    sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    sitk_img = sitk.JoinSeries(sitk_img)

    mask_mod = mask / 255 + 1
    sitk_mask = sitk.GetImageFromArray(mask_mod)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)

    # Parameters for radiomics extractor
    params = {}
    params['binWidth'] = 20
    params['sigma'] = [1, 2, 3]
    params['verbose'] = True

    # For GLRLM features
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glrlm')

    result = extractor.execute(sitk_img, sitk_mask)
    glrlm_features_list = []
    for key, value in result.items():
        if 'glrlm' in key:
            glrlm_features_list.append(value.item())

    glrlm_features = np.array(glrlm_features_list)

    masked_roi = bring_to_256_levels(np.multiply(roi_gray, mask))
    try:
        textures = mt.features.haralick(masked_roi, ignore_zeros=True)
        haralick_features = textures.mean(axis=0)
    except:
        haralick_features = np.zeros((13,))

    texture_features = np.concatenate((haralick_features, glrlm_features), axis=0)

    return texture_features


def get_asymmetry():
    pass


def feature_hu_moments(contour):
    """
    Calculate the shape features based on HU moments

    Parameters
    ----------
    bin_roi     binary version of the ROI

    Returns
    -------
    hu_moments  numpy array containing the HU moments

    """

    hu_moments = cv.HuMoments(cv.moments(contour))
    # Log scale transform
    for i in range(0, 7):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments.reshape(-1)


def multi_scale_lbp_features(roi):
    """
    Calculate multi scale local binary pattern based on mahotas
    Parameters
    ----------
    roi:        numpy array of The region of interest(8-bits)

    Returns
    -------
    lbp         Histogram of multi scale lbp
    """
    roi_img = cv.normalize(roi, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    roi_or = np.copy(roi_img)
    r = 1
    R = 1
    i = 0
    lbp = np.zeros((5, 36))
    while R < 35:
        lb_p = mt.features.lbp(roi_img, np.rint(R), 8, ignore_zeros=False)
        lb_p = lb_p / np.sum(lb_p)
        lbp[i, :] = lb_p
        r_1 = r
        r = r_1 * (2 / (1 - np.sin(np.pi / 8)) - 1)
        R = (r + r_1) / 2
        if np.floor(r - r_1) % 2 == 0:
            k_size = np.int(np.ceil(r - r_1))
        else:
            k_size = np.int(np.floor(r - r_1))
        std_dev = (k_size / 2)
        roi_img = cv.GaussianBlur(roi_or, (k_size, k_size), std_dev)
        i += 1

    return lbp.reshape((180,))


def features_hog(roi):
    """
    Calculate a scale based Histogram of Oriented gradients based on ski-image

    Parameters
    ----------
    roi             numpy array of The region of interest(8-bits)

    Returns
    -------
    hog features    numpy array containing a HOG descriptor of the image

    """
    width = np.int(roi.shape[0] / 10)
    height = np.int(roi.shape[1] / 10)
    w_t = np.int((roi.shape[0] - width * 10) / 2)
    h_t = np.int((roi.shape[1] - height * 10) / 2)
    crop_roi = roi[w_t: w_t + 10 * width, h_t: h_t + 10 * height]
    f_hog = hog(crop_roi, orientations=8, pixels_per_cell=(width, height),
                cells_per_block=(1, 1), visualize=False, multichannel=False)
    return f_hog


def extract_features(roi_color, hog=False, haralick="full", lbp_type="multi"):
    """
    Extract all the features of a ROI. A total of 41 features are extracted. LBP and HoG are not activated

    Parameters
    ----------
    roi_color           Region of interest of the image (RGB)
    contour             Contour containing the ROI
    mask                Binary version of the ROI

    Returns
    -------
    feature_vector      All extracted features of a ROI
    """

    roi_gray = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
    mask = roi = np.ones_like(roi_gray)

    # Color based features
    color_features = get_color_based_features(roi_color)

    # LBP
    lbp = []
    if lbp_type == "multi":
        lbp = multi_scale_lbp_features(roi_gray)
    elif lbp_type == "single":
        lbp = mt.features.lbp(roi_gray, 1, 8, ignore_zeros=False)
        lbp = lbp.T

    # Texture: Haralick features using Mahotas (vector 1x13)
    texture_features = []
    if haralick == "full":
        texture_features = get_texture_features(roi_gray, mask)
    elif haralick == "mh":
        texture_features = mt.features.haralick(roi_gray, ignore_zeros=True)
        texture_features = texture_features.flatten()
    # HOG features
    hog_features = []
    if hog is True:
        hog_features = features_hog(roi_gray)

    return np.transpose(np.concatenate((texture_features, lbp, color_features.T, hog_features), axis=0))


def extract_daisy(roi_color):
    """
    Extract daisy features
    ****NOT USED
    Parameters
    ----------
    roi_color       color images

    Returns         daisy features
    -------

    """
    roi_gray = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
    descs = daisy(roi_gray, step=8, radius=16, rings=2, histograms=6,
                  orientations=8, visualize=False)
    return descs.flatten()


def extract_surf(roi_color):
    """
        Extract surf features
        ****NOT USED
        Parameters
        ----------
        roi_color       color images

        Returns         sirf features for each keypoint detected
        -------

        """

    gray = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
    n_kp = 200
    sift = cv.xfeatures2d.SIFT_create(n_kp)
    _, des = sift.detectAndCompute(gray, None)
    descriptor = np.zeros((n_kp, 128), dtype=np.uint8)
    if des is not None:
        des = des.astype(np.uint8)
        if len(des) > n_kp:
            descriptor = des[:n_kp, :]
        else:
            descriptor[:des.shape[0], :] = des

    return descriptor.flatten()


def extract_dense_features(roi_color, step_x, step_y=None):
    """
        Extract daisy features
        ****NOT USED
        Parameters
        ----------
        roi_color       color images
        step_x          step in x to grid features
        tep_y=None      step in y for grid features

        Returns         ddense extraction of features
        -------

        """
    shape_y = roi_color.shape[0]
    shape_x = roi_color.shape[1]

    if step_y is None:
        step_y = step_x

    n_x = np.int(shape_x / step_x)
    n_y = np.int(shape_y / step_y)
    x_samples = np.linspace(0, n_x * step_x, num=n_x + 1).astype(np.int)
    y_samples = np.linspace(0, n_y * step_y, num=n_y + 1).astype(np.int)

    features_regions = None

    for i_x in range(n_x):
        for i_y in range(n_y):
            x_low = x_samples[i_x]
            x_high = x_samples[i_x + 1]
            y_low = y_samples[i_y]
            y_high = y_samples[i_y + 1]

            roi = roi_color[x_low:x_high, y_low:y_high, :]
            features = extract_features(roi, haralick="full", lbp_type="single")
            if features_regions is None:
                features_regions = np.zeros((n_x * n_y, len(features)))

            pos = i_y + i_x * n_y
            features_regions[pos, :] = features

    full_set = features_regions.flatten()
    return full_set
