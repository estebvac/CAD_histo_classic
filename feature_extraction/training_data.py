from os.path import join
import pandas as pd
from main_flow.flow import process_single_image
from .build_features_file import read_images
from tqdm import tqdm
import time
import numpy as np


def __get_features(path, full_images_df, debug=False):
    """
    calculate the features of all the images of the dataset
    Parameters
    ----------
    full_images_df  Dataframe containing the path and class of all the images

    Returns
    -------
    total_features  feature vector of all the ROIs in the image

    """
    total_images = len(full_images_df)
    meta_data = full_images_df
    total_features = []
    pbar = tqdm(range(0, total_images))
    for img_counter in pbar:
        file = full_images_df['File'][img_counter]
        folder = full_images_df['Class'][img_counter]
        img_path = path + '/' + folder + '/' + file
        features = process_single_image(img_path, debug)
        meta_data['File'][img_counter] = meta_data['File'][img_counter].split('/')[-1]
        if img_counter == 0:
            total_features_0 = features.reshape((1,-1))
            total_features = np.repeat(total_features_0, total_images, axis=0)

        total_features[img_counter, :] = features

        time.sleep(0.0001)
    pbar.close()
    pbar.clear()

    return total_features, meta_data


def prepate_datasets(dataset_path, output_name, debug=False):
    """
    Prepare the feature extraction of all the dataset

    Parameters
    ----------
    dataset_path:   path to the full dataset
    output_name:    Name of the csv file to save
    debug:          Show the image output

    Returns
    -------
    CSV files:      dataframe containing the features and dataframe containing metadata of the ROIs

    """

    full_images_df = read_images(dataset_path)
    print("Preparing " + output_name + " set!\n")
    total_features, meta_data = __get_features(dataset_path, full_images_df, debug)
    meta_data.to_csv(join(dataset_path, output_name + ".csv"))
    numpy_path = join(dataset_path, output_name + ".npy")
    np.save(numpy_path, total_features)
    #return training_features
