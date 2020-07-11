import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

CONFIG = {
    "n": 100,
    "jobs": -1,
    "seed": 7,
    "k-Fold": 5,
    "test_size": 0.25
}

PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PATH, os.pardir, os.pardir, "data")
DATA_PATH = os.path.join(DATA, "training_data")
META_DATA = os.path.join(DATA, "meta_data")
LABEL = "M_STAGE"

# Select data (According to precision_recall(_cv) graph on meta_data)
clinical_features = pd.read_csv(os.path.join(DATA_PATH, "clinical_data", "encoded_clinical_data.tsv"),
                                sep="\t", index_col=0)
clinical_features_tml = pd.read_csv(os.path.join(DATA_PATH, "merged_data", "clinical_data_tml.tsv"),
                                    sep="\t", index_col=0)
selected_data = [
    (clinical_features, "CF"),
    (clinical_features_tml, "CF + TML")
]


def abstract_resample(x_train: pd.DataFrame, y_train: pd.Series,
                      column_name: str) -> (pd.DataFrame, pd.DataFrame):
    """
    First part of resample

    :param x_train: x_train data
    :param y_train: y_train data
    :param column_name: Name of column of class under-represented
    :return: A tuple with: input_data (pandas.DataFrame) and
        labels (pd.Series) resampled
    """
    whole = pd.concat([x_train, y_train], axis=1)
    label_names = list(sorted(pd.unique(y_train)))
    class_1 = whole[whole[column_name] == label_names[0]]
    class_2 = whole[whole[column_name] == label_names[1]]
    if len(class_1) < len(class_2):
        under = class_1
        over = class_2
    else:
        under = class_2
        over = class_1
    return under, over


def oversample(x_train: pd.DataFrame, y_train: pd.Series,
               column_name: str) -> (pd.DataFrame, pd.Series):
    """
    Oversamples based on under represented class

    :param x_train: x_train data
    :param y_train: y_train data
    :param column_name: Name of column of class under-represented
    :return: A tuple with: input_data (pandas.DataFrame) and
        labels (pd.Series) resampled
    """
    under, over = abstract_resample(x_train, y_train, column_name)
    class_oversampled = resample(under,
                                 replace=True,
                                 n_samples=len(over),
                                 random_state=CONFIG["seed"])
    x_new = pd.concat([class_oversampled, over])
    x_train = x_new.drop(columns=[column_name])
    y_train = x_new[column_name]
    return x_train, y_train


def undersample(x_train: pd.DataFrame, y_train: pd.Series,
                column_name: str) -> (pd.DataFrame, pd.Series):
    """
    Undersamples based on under represented class

    :param x_train: x_train data
    :param y_train: y_train data
    :param column_name: Name of column of class under-represented
    :return: A tuple with: input_data (pandas.DataFrame) and
        labels (pd.Series) resampled
    """
    under, over = abstract_resample(x_train, y_train, column_name)
    class_undersampled = resample(over,
                                  replace=True,
                                  n_samples=under,
                                  random_state=CONFIG["seed"])
    x_new = pd.concat([class_undersampled, under])
    x_train = x_new.drop(columns=[column_name])
    y_train = x_new[column_name]
    return x_train, y_train


def important_feature(x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    :param x_train: x_train to fit model
    :param y_train: y_train to fit model
    :return: Dictionary: feature / value
    """
    output = dict()
    random_forest = RandomForestClassifier(n_estimators=CONFIG["n"],
                                           n_jobs=CONFIG["jobs"],
                                           random_state=CONFIG["seed"])
    random_forest.fit(x_train, y_train)
    for label, feature in zip(x_train.columns, random_forest.feature_importances_):
        output[label] = feature
    return output


if __name__ == "__main__":
    for data, name in selected_data:
        print("\tUsing {} to classify".format(name))
        input_data = data.drop(columns=LABEL)
        labels = data[LABEL]
        print("\t\tResample: {}".format("No resample"))
        no_resample = important_feature(input_data, labels)
        print("\t\tResample: {}".format("Undersample"))
        under_sample = important_feature(*undersample(input_data, labels, LABEL))
        print("\t\tResample: {}".format("Oversample"))
        over_sample = important_feature(*oversample(input_data, labels, LABEL))
        pd.DataFrame({
            "No Resample": no_resample,
            "Undersample": under_sample,
            "Oversample": over_sample
        }).to_csv(os.path.join(META_DATA, "feature_importances_{}.csv".format(name)), sep=",")
