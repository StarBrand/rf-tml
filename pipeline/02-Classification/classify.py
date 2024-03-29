import os
import pandas as pd
from useful_methods import pca
from models import RandomForest

PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PATH, os.pardir, os.pardir, "data", "training_data")
LABEL = "M_STAGE"


if __name__ == "__main__":
    for folder in os.listdir(DATA_PATH):
        print("Using folder {}".format(folder))
        for file in os.listdir(os.path.join(DATA_PATH, folder)):
            name, format_file = file.split(".")
            if format_file == "tsv":
                print("\tUsing {} to classify".format(name))
                data = pd.read_csv(os.path.join(DATA_PATH, folder, file),
                                   sep="\t", index_col=0)
                input_data = data.drop(columns=LABEL)
                labels = data[LABEL]
                try:
                    for pca_on in [False, True]:
                        pca_log = "No PCA"
                        if pca_on:
                            try:
                                input_data = pca(input_data)
                                name += "_pca"
                                pca_log = "With PCA"
                            except Exception as e:
                                print("\t\tPCA failed:\t{}".format(e))
                                continue
                        random_forest = RandomForest(name)
                        for validation, val_name in [("split", "Hold-out"), ("cv", "5-Fold Cross Validation")]:
                            for resampling, re_name in [("none", "No resample"),
                                                        ("under", "undersampling"),
                                                        ("over", "oversampling")]:
                                print(
                                    "\t\tClassifying using => validation: {val}\tresample: {re}\tpca: {pca}".format(
                                        val=val_name, re=re_name, pca=pca_log
                                    )
                                )
                                random_forest.execute_test(
                                    input_data, labels, validation=validation, resampling=resampling
                                )
                except ValueError as e:
                    # Not use clinical data (not encoded it will not work)
                    print("\t\t Cannot use: {}".format(e))
