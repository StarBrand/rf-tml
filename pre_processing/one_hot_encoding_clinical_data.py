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
aggregated_data = pd.read_csv(os.path.join(DATA, "clinical_data_aggregated.csv"), sep="\t",
                              index_col="SAMPLE_ID")
no_stage_data = pd.read_csv(os.path.join(DATA, "clinical_data_no_tumor_stage.csv"), sep="\t",
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


def encode_sex(clinical_data: pd.DataFrame) -> None:
    """
    Encoding sex column on clinical_data

    :param clinical_data: Data to encode
    :return: None
    """
    sex = encoder.fit_transform(pd.DataFrame(clinical_data["SEX"]).replace(np.nan, "None")).toarray()
    clinical_data.insert(0, "female", sex[:, 0], True)
    clinical_data.insert(1, "male", sex[:, 1], True)
    clinical_data.drop(columns=["SEX"], inplace=True)
    return


def encode_smoking_h(clinical_data: pd.DataFrame) -> str:
    """
    Encoding smoking history column on clinical_data

    :param clinical_data: Data to encode
    :return: Return detail of encoded categorizes
    """
    smoking_h = encoder.fit_transform(pd.DataFrame(clinical_data["SMOKING_HISTORY"]).replace(np.nan, "None")).toarray()
    detail = "Smoking History Categories:\n"
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            detail += "\tcat{}: {}\n".format(i, cat)
            clinical_data.insert(4 + i, "smoke_h_cat_{}".format(i), smoking_h[:, i], True)
    clinical_data.drop(columns=["SMOKING_HISTORY"], inplace=True)
    return detail


def encode_tumor_stage(clinical_data: pd.DataFrame, detail: str) -> str:
    """
    Encoding smoking history column on clinical_data

    :param clinical_data: Data to encode
    :param detail: String with encoded categorizes
    :return: Return detail updated
    """
    stage = encoder.fit_transform(pd.DataFrame(clinical_data["STAGE"]).replace(np.nan, "None")).toarray()
    detail += "Stage Categories:\n"
    for i, cat in enumerate(encoder.categories_[0]):
        if cat != "None":
            detail += "\tcat{}: {}\n".format(i, cat)
            clinical_data.insert(4 + i, "stage_cat_{}".format(i), stage[:, i], True)
    clinical_data.drop(columns=["STAGE"], inplace=True)
    return detail


if __name__ == '__main__':
    # Age
    encode_age(data)
    encode_age(aggregated_data)
    encode_age(no_stage_data)

    # Sex
    encode_sex(data)
    encode_sex(aggregated_data)
    encode_sex(no_stage_data)

    # Smoking History
    details = encode_smoking_h(data)
    aggregated_details = encode_smoking_h(aggregated_data)
    no_stage_details = encode_smoking_h(no_stage_data)

    # Stage
    details = encode_tumor_stage(data, details)
    aggregated_details = encode_tumor_stage(aggregated_data, aggregated_details)

    for file_name, content in zip(["Encoded_Clinical_Data",
                                   "Encoded_Aggregated_Data",
                                   "Encoded_No_Stage_Data"],
                                  [details, aggregated_details, no_stage_details]):
        with open(
            os.path.join(DATA, "{}.txt".format(file_name)),
            "w", encoding="utf-8"
        ) as file:
            file.write(content)

    data.dropna(inplace=True)
    aggregated_data.dropna(inplace=True)
    no_stage_data.dropna(inplace=True)

    # Done
    data.to_csv(os.path.join(DATA, "encoded_clinical_data.csv"), "\t")
    aggregated_data.to_csv(os.path.join(DATA, "encoded_clinical_data_aggregated.csv"), "\t")
    no_stage_data.to_csv(os.path.join(DATA, "encoded_clinical_data_no_tumor_stage.csv"), "\t")
