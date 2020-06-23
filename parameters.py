PARAMETERS = {
    "file": "encoded_clinical_data",  # Name of file to use as input data
    "file_type": "csv",  # Type of file csv, txt, etc
    "pca": True,  # Whether or not do a pca
    "labels": "M_STAGE",  # Label of labels to classify
    "columns_to_drop": ["M_STAGE"],  # Columns to drop on input
    "name": "clinical_data_pca_oversample",  # Name to be used to save results
    "option": "over"  # split, under, over, cv
}
