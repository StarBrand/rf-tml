"""one_hot_encoding_clinical_data.py: One-hot encoding of
categorical data on 'clinical_data.csv'"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

DATA = os.path.join(os.pardir, "data", "clinical_data")

encoder = OneHotEncoder(handle_unknown="ignore")
data = pd.read_csv(os.path.join(DATA, "clinical_data.csv"), sep="\t",
                   index_col="SAMPLE_ID")
details = ""

if __name__ == '__main__':
    # Age
    data.insert(2, "older", 1.0 * (data["AGE"] >= 60), True)
    data.insert(3, "younger", 1.0 * (data["AGE"] < 60), True)
    data.drop(columns=["AGE"], inplace=True)

    # Sex
    sex = encoder.fit_transform(pd.DataFrame(data["SEX"]).replace(np.nan, "None")).toarray()
    data.insert(0, "female", sex[:, 0], True)
    data.insert(1, "male", sex[:, 1], True)
    data.drop(columns=["SEX"], inplace=True)

    # Smoking History
    smoking_h = encoder.fit_transform(pd.DataFrame(data["SMOKING_HISTORY"]).replace(np.nan, "None")).toarray()
    details += "Smoking History Categories:\n"
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            details += "\tcat{}: {}\n".format(i, cat)
            data.insert(4 + i, "smoke_h_cat_{}".format(i), smoking_h[:, i], True)
    data.drop(columns=["SMOKING_HISTORY"], inplace=True)

    # Stage
    stage = encoder.fit_transform(pd.DataFrame(data["STAGE"]).replace(np.nan, "None")).toarray()
    details += "Stage Categories:\n"
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            details += "\tcat{}: {}\n".format(i, cat)
            data.insert(4 + i, "stage_cat_{}".format(i), stage[:, i], True)
    data.drop(columns=["STAGE"], inplace=True)

    with open(
            os.path.join(DATA, "Encoded_Clinical_Data.txt"),
            "w", encoding="utf-8"
    ) as file:
        file.write(details)

    data.dropna(inplace=True)

    # Done
    data.to_csv(os.path.join(DATA, "encoded_clinical_data.csv"), "\t")
