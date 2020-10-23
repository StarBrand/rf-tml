"""pca.py: PCA function"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from useful_methods import CONFIG


def pca(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Does a PCA to the data

    :param raw_data: Data to do a PCA
    :return: An array with input data
    """
    index = raw_data.index
    normalized_data = StandardScaler().fit_transform(raw_data)
    return pd.DataFrame(
        PCA("mle", random_state=CONFIG["seed"]).fit_transform(normalized_data),
        index=index
    )
