"""resample.py: Resampling functions"""

import pandas as pd
from sklearn.utils import resample
from useful_methods import CONFIG


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
                                  n_samples=len(under),
                                  random_state=CONFIG["seed"])
    x_new = pd.concat([class_undersampled, under])
    x_train = x_new.drop(columns=[column_name])
    y_train = x_new[column_name]
    return x_train, y_train
