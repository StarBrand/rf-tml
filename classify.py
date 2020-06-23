import os
import pandas as pd
from sklearn.decomposition import PCA
from parameters import PARAMETERS
from models import RandomForest, CONFIG

data = pd.read_csv(os.path.join("data", "{}.{}".format(PARAMETERS["file"],
                                                        PARAMETERS["file_type"])),
                   sep="\t", index_col=0)

input_data = data.drop(columns=PARAMETERS["columns_to_drop"])
labels = data[PARAMETERS["labels"]]


def pca(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Does a PCA to the data

    :param raw_data: Data to do a PCA
    :return: An array with input data
    """
    index = raw_data.index
    return pd.DataFrame(
        PCA("mle", random_state=CONFIG["seed"]).fit_transform(raw_data),
        index=index
    )


if __name__ == "__main__":
    if PARAMETERS["pca"]:
        input_data = pca(input_data)
    random_forest = RandomForest(input_data, labels)
    random_forest.execute_test(
        PARAMETERS["name"],
        PARAMETERS["option"])
