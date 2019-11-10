import pandas as pd
import os
from feature_extraction import utils
import numpy as np


def read_images(general_path):
    """
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    """

    # benign and malignant
    all_files = []
    all_folder = []
    for folder in utils.get_dirs(general_path):
        files = utils.get_files(os.path.join(general_path, folder))
        all_files += files
        folders = [folder ]* len(files)
        all_folder += folders

    all_files = np.asarray(all_files).reshape((-1,1))
    all_folder = np.asarray(all_folder).reshape((-1, 1))
    data_tuple = np.concatenate((all_files,all_folder), axis=1)
    images_df = pd.DataFrame(data_tuple, columns=['File', 'Class'])






    return images_df
