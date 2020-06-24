PARAMETERS = {
    "folder": "merged_data",  # Name of folder to look data
    "file": "clinical_data_mut_not_norm",  # Name of file to use as input data
    "pca": False,  # Whether or not do a pca
    "labels": "M_STAGE",  # Label of labels to classify
    "columns_to_drop": ["M_STAGE"],  # Columns to drop on input
    "name": "clinical_data_mut_not_norm",  # Name to be used to save results
    "option": "cv"  # split, under, over, cv
}
