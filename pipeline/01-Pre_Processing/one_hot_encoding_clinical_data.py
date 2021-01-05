"""one_hot_encoding_clinical_data.py: One-hot encoding of
categorical data on 'clinical_data.csv'"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "training_data", "clinical_data")

encoder = OneHotEncoder(handle_unknown="ignore")
data = pd.read_csv(os.path.join(DATA, "clinical_data.tsv"), sep="\t",
                   index_col="SAMPLE_ID")
no_stage_data = pd.read_csv(os.path.join(DATA, "clinical_data_no_tumor_stage.tsv"), sep="\t",
                            index_col="SAMPLE_ID")


def encode_age(clinical_data: pd.DataFrame) -> None:
    """
    One hot encoding age column on clinical_data

    :param clinical_data: Clinical data to encode
    :return: None, alter reference to clinical data
    """
    clinical_data.insert(2, "older", 1.0 * (clinical_data["AGE"] >= 60), True)
    clinical_data.insert(3, "younger", 1.0 * (clinical_data["AGE"] < 60), True)
    clinical_data.drop(columns=["AGE"], inplace=True)
    return


def encode_smoking_h(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding smoking history column on clinical_data

    :param clinical_data: Data to encode
    :return: Return detail of encoded categorizes
    """
    smoking_h = encoder.fit_transform(pd.DataFrame(clinical_data["SMOKING_HISTORY"]).replace(np.nan, "None")).toarray()
    detail = pd.DataFrame({}, columns=["Attribute", "Category", "Encoding name"])
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            detail = detail.append({"Attribute": "Smoking History",
                                    "Category": cat,
                                    "Encoding name": "smoke_h_cat_{}".format(i)}, ignore_index=True)
            clinical_data.insert(3 + i, "smoke_h_cat_{}".format(i), smoking_h[:, i], True)
    clinical_data.drop(columns=["SMOKING_HISTORY"], inplace=True)
    return detail


def encode_cancer_type(clinical_data: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding cancer type column on clinical_data

    :param clinical_data: Data to encode
    :param detail: String with encoded categorizes
    :return: Return detail updated
    """
    stage = encoder.fit_transform(pd.DataFrame(clinical_data["CANCER_TYPE_DETAILED"]).replace(np.nan, "None")).toarray()
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            detail = detail.append({"Attribute": "Cancer Type",
                                    "Category": cat,
                                    "Encoding name": "cancer_type_{}".format(i)}, ignore_index=True)
            clinical_data.insert(6 + i, "cancer_type_{}".format(i), stage[:, i], True)
    clinical_data.drop(columns=["CANCER_TYPE_DETAILED"], inplace=True)
    return detail


def encode_tumor_stage(clinical_data: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding tumor stage column on clinical_data

    :param clinical_data: Data to encode
    :param detail: String with encoded categorizes
    :return: Return detail updated
    """
    stage = encoder.fit_transform(pd.DataFrame(clinical_data["STAGE"]).replace(np.nan, "None")).toarray()
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            detail = detail.append({"Attribute": "Stage",
                                    "Category": cat,
                                    "Encoding name": "stage_cat_{}".format(i)}, ignore_index=True)
            clinical_data.insert(8 + i, "stage_cat_{}".format(i), stage[:, i], True)
    clinical_data.drop(columns=["STAGE"], inplace=True)
    return detail


if __name__ == '__main__':
    # Age
    encode_age(data)
    encode_age(no_stage_data)

    # Smoking History
    details = encode_smoking_h(data)
    no_stage_details = encode_smoking_h(no_stage_data)

    # Cancer type
    details = encode_cancer_type(data, details)
    no_stage_details = encode_cancer_type(no_stage_data, no_stage_details)

    # Stage
    details = encode_tumor_stage(data, details)

    details.to_csv(os.path.join(DATA, "Encoded_Clinical_Data.txt"), sep="\t", index=False)
    no_stage_details.to_csv(os.path.join(DATA, "Encoded_No_Stage_Data.txt"), sep="\t", index=False)

    data.dropna(inplace=True)
    no_stage_data.dropna(inplace=True)

    # Done
    data.to_csv(os.path.join(DATA, "encoded_clinical_data.tsv"), "\t")
    no_stage_data.to_csv(os.path.join(DATA, "encoded_clinical_data_no_tumor_stage.tsv"), "\t")
