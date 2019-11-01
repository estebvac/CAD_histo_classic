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
    #total_images = 10
    total_features = []
    for img_counter in tqdm(range(0, total_images)):
        features = process_single_image(full_images_df['File'][img_counter], debug)

        features.insert(0, "label", full_images_df['Class'][img_counter], True)

        img_name = full_images_df['File'][img_counter].split('/')[-1]
        features.insert(0, "Name", img_name, True)
        if img_counter == 0:
            total_features = features.to_numpy()#np.zeros((total_images, features.size))
            total_features = np.repeat(total_features, total_images, axis=0)

        total_features[img_counter, :] = features.to_numpy().reshape(-1)

        time.sleep(0.0001)

    df = pd.DataFrame(total_features, columns=features.columns)
    return [df]


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
    print("Preparing training set!\n")
    [training_features] = __get_features(dataset_path, full_images_df, debug)
    training_features.to_csv(join(dataset_path, output_name))
    return training_features
