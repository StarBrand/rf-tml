"""model.py: Model abstract class"""

import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, KFold

from models import CONFIG
from useful_methods import oversample, undersample

PATH = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(*([PATH] + [os.pardir] * 2 + ["data"]))
METRICS = os.path.join(DATA, "metrics")
os.makedirs(METRICS, exist_ok=True)

matplotlib.use('Agg')

TYPE_OF_TEST = {
    "cv": "{}-Fold_CrossValidation".format(CONFIG["k-Fold"]),
    "split": "Hold-out"
}
RESAMPLE = {
    "under": "Undersample",
    "over": "Oversample",
    "none": "NoResample"
}


class Model(ABC):
    """
    Model Class

    :ivar classifier: A Classifier Mixin or any other
    :ivar name: Name to identify test
    """

    @abstractmethod
    def __init__(self, name: str) -> None:
        """
        Generates instance of Random Forest

        :param name: Name to be used to report result and metrics
        :return:
        """
        self.classifier = None
        self.name = name
        return

    def execute_test(self, input_data: pd.DataFrame,
                     labels: pd.Series, resampling: str = "none",
                     validation: str = "split") -> None:
        """
        Classifies from data as it is

        :param input_data: Input data to base classification
        :param labels: Labeled results to predict
        :param resampling: Use all data ("none") or resampling by using oversample ("over")
            or undersample ("under")
        :param validation: Validate model using Hold-out ("split") or using
            cross validation ("cv")
        :return: None, generate output
        """
        label_names = list(sorted(pd.unique(labels)))
        if validation == "cv":
            c_m = self.__cross_validation_test(input_data, labels, label_names, resampling)
        elif validation == "split":
            x_train, x_test, y_train, y_test = train_test_split(input_data, labels,
                                                                test_size=CONFIG["test_size"],
                                                                random_state=CONFIG["seed"])
            c_m = self.__get_confusion_matrix(x_train, x_test, y_train, y_test, label_names, resampling)
        else:
            sys.exit("Not a valid option")
        self.__save_results(c_m, label_names, validation, resampling)

    @staticmethod
    def __resample(data: pd.DataFrame, labels: pd.Series, column_name: str,
                   resampling: str) -> (pd.DataFrame, pd.Series):
        """
        Re-samples data by "resampling" (over or undersampling)
        based on under represented class

        :param data: Data to be resampled (train set)
        :param labels: Name of labels (to be predicted)
        :param column_name: Name of column of class under-represented
        :param resampling: Sampling method: "over" or "under"
        :return: A tuple with: input_data (pandas.DataFrame) and
            labels (pd.Series) resampled
        """
        # Assuming 2 classes
        if resampling == "over":
            return oversample(data, labels, column_name)
        elif resampling == "under":
            return undersample(data, labels, column_name)
        else:
            return data, labels

    def train_test(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series,
                   resampling: str) -> (pd.Series, pd.Series):
        """
        Classifies using train/test

        :param x_train: Data to train
        :param x_test: Data to test
        :param y_train: Categories to train
        :param y_test: Categories for testing
        :param resampling: Data as it is ("none"), undersample ("under") or oversample ("over")
        :return: Test label and predict label
        """
        if resampling != "none":
            column_name = str(y_train.name)
            x_train, y_train = self.__resample(x_train, y_train, column_name, resampling)
        classifier = deepcopy(self.classifier)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        return y_test, y_pred

    def __get_confusion_matrix(self, x_train: pd.DataFrame, x_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series,
                               label_names: [str], resampling: str) -> np.ndarray:
        """
        Classifies using train/test and return confusion matrix got

        :param x_train: Data to train
        :param x_test: Data to test
        :param y_train: Categories to train
        :param y_test: Categories for testing
        :param label_names: Name of labels
        :param resampling: Data as it is ("none"), undersample ("under") or oversample ("over")
        :return: Test label and predict label
        """
        y_test, y_pred = self.train_test(x_train, x_test, y_train, y_test, resampling)
        return confusion_matrix(y_test, y_pred, labels=label_names)

    def __cross_validation_test(self, input_data: pd.DataFrame, labels: pd.Series,
                                label_names: [str], resampling: str) -> np.ndarray:
        """
        Classifies using cross validation

        :param input_data: Data to use
        :param labels: Labeled categorize
        :param label_names: Name of labels (unique)
        :return: None, generate output
        """
        index_generator = KFold(
            n_splits=CONFIG["k-Fold"],
            shuffle=True,
            random_state=CONFIG["seed"]
        ).split(input_data, labels)
        c_m = None
        for train_indexes, test_index in index_generator:
            x_train = input_data.iloc[train_indexes]
            y_train = labels.iloc[train_indexes]
            x_test = input_data.iloc[test_index]
            y_test = labels.iloc[test_index]
            partial_c_m = self.__get_confusion_matrix(x_train, x_test, y_train, y_test, label_names, resampling)
            if c_m is None:
                c_m = partial_c_m
            else:
                c_m += partial_c_m
        return c_m

    def __save_results(self, c_m: np.ndarray, labels: [str], validation: str, resampling: str) -> None:
        """
        Save results

        :param c_m: Confusion matrix got
        :param labels: Labels name
        :param validation: By using cross validation ("cv") or hold-out ("split")
        :param resampling: Resampling method ("under", "over", "none")
        :return: None, generate output
        """
        c_m_display = ConfusionMatrixDisplay(c_m)
        c_m_display.plot()
        os.makedirs(os.path.join(METRICS, self.name), exist_ok=True)
        folder = os.path.join(METRICS, self.name, TYPE_OF_TEST[validation])
        os.makedirs(folder, exist_ok=True)
        folder = os.path.join(folder, RESAMPLE[resampling])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, "confusion_matrix.png"))
        plt.clf()
        plt.close(c_m_display.figure_)
        accuracy_score = accuracy(c_m)
        precision_score = precision(c_m)
        recall_score = recall(c_m)
        f1_score_value = f1_score(c_m)
        with open(os.path.join(folder, "scores.csv"), "w", encoding="utf-8") as file:
            file.write(
                "Accuracy,{acc}\n"
                "{label1}_Precision,{pre_1}\n"
                "{label1}_Recall,{rec_1}\n{label1}_F1-score,{f1_1}\n"
                "{label2}_Precision,{pre_2}\n"
                "{label2}_Recall,{rec_2}\n{label2}_F1-score,{f1_2}\n".format(
                    label1=labels[0], acc=accuracy_score,
                    pre_1=precision_score[0],
                    rec_1=recall_score[0],
                    f1_1=f1_score_value[0],
                    label2=labels[1], pre_2=precision_score[1],
                    rec_2=recall_score[1],
                    f1_2=f1_score_value[1])
            )


def accuracy(matrix: np.ndarray) -> float:
    """
    Calculate accuracy of prediction
    :param matrix: Confusion matrix calculated
    :return: Accuracy
    """
    return matrix.diagonal().sum() / matrix.sum()


def precision(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate precision of prediction
    :param matrix: Confusion matrix
    :return: Precision for every class
    """
    return np.nan_to_num(np.divide(matrix.diagonal(), matrix.sum(axis=0)))


def recall(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate recall of prediction
    :param matrix: Confusion matrix
    :return: Recall for every class
    """
    return np.nan_to_num(np.divide(matrix.diagonal(), matrix.sum(axis=1)))


def f1_score(matrix: np.ndarray) -> np.ndarray:
    """
    F1-score of prediction
    :param matrix: Confusion matrix
    :return: F1-score for class
    """
    a = precision(matrix)
    b = recall(matrix)
    return 2 * np.nan_to_num(np.divide(np.multiply(a, b), a + b))
