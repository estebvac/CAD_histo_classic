from feature_extraction.training_data import prepate_datasets

path_train = r"C:\Users\esteb\Documents\CAD_PROJECT\HISTO\train"
path_val = r"C:\Users\esteb\Documents\CAD_PROJECT\HISTO\val"
path_test = r"C:\Users\esteb\Documents\CAD_PROJECT\HISTO\test"

#prepate_datasets(path_train, "train")
#prepate_datasets(path_val, "val")
prepate_datasets(path_test, "test")