"""random_forest.py: Random Forest training and prediction"""

import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from models.random_forest import CONFIG
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

matplotlib.use('Agg')

FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(FOLDER, os.pardir, os.pardir, os.pardir, "results")
TYPE_OF_TEST = {
    "cv": "{}-Fold_CrossValidation".format(CONFIG["k-Fold"]),
    "split": "Test_Set",
    "under": "Undersample",
    "over": "Oversample"
}


class RandomForest:
    """
    Random Forest Class

    :ivar classifier: A RandomForestClassifier
    :ivar data: Data to use as input
    :ivar labels: Labels to classify
    """

    def __init__(self, data: pd.DataFrame, labels: pd.Series) -> None:
        """
        Generates instance of Random Forest

        :param data: Data to use as predictors
        :param labels: Labels to classify
        :return:
        """
        self.classifier = RandomForestClassifier(n_estimators=CONFIG["n"],
                                                 n_jobs=CONFIG["jobs"],
                                                 random_state=CONFIG["seed"])
        self.data = data
        self.labels = labels

    def execute_test(self, name: str, option: str) -> None:
        """
        Classifies from data as it is

        :param name: Name of test to save
        :param option: By using cross validation ("cv"), train-test as it is ("split"),
            undersample ("under") or oversample ("over")
        :return: None, generate output
        """
        if option == "cv":
            self.__cross_validation_test(name)
        elif option in ["split", "under", "over"]:
            self.__train_test(name, option)
        else:
            sys.exit("Not a valid option")

    def __train_test(self, name: str, option: str) -> None:
        """
        Classifies using train/test

        :param name: Name of test to save
        :param option: Data as it is ("split"),
            undersample ("under") or oversample ("over")
        :return: None, generate output
        """
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels,
                                                            test_size=CONFIG["test_size"],
                                                            random_state=CONFIG["seed"])
        labels = pd.unique(y_train)
        if option in ["under", "over"]:
            whole = pd.concat([x_train, y_train], axis=1)
            class_name = y_train.name
            # Assuming 2 classes
            class_1 = whole[whole[class_name] == labels[0]]
            class_2 = whole[whole[class_name] == labels[1]]
            if len(class_1) < len(class_2):
                under = class_1
                over = class_2
            else:
                under = class_2
                over = class_1
            if option == "under":
                class_undersampled = resample(over,
                                              replace=True,
                                              n_samples=len(under),
                                              random_state=CONFIG["seed"])
                x_new = pd.concat([class_undersampled, under])
            else:  # option == "over"
                class_oversampled = resample(under,
                                             replace=True,
                                             n_samples=len(over),
                                             random_state=CONFIG["seed"])
                x_new = pd.concat([class_oversampled, over])
            x_train = x_new.drop(columns=[class_name])
            y_train = x_new[class_name]
        classifier = deepcopy(self.classifier)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        self.__save_results(name, y_test, y_pred, labels, option)

    def __cross_validation_test(self, name: str) -> None:
        """
        Classifies using cross validation

        :param name: Name of test to save
        :return: None, generate output
        """
        y_pred = cross_val_predict(self.classifier, self.data, self.labels,
                                   cv=CONFIG["k-Fold"], n_jobs=CONFIG["jobs"])
        labels = pd.unique(self.labels)
        self.__save_results(name, self.labels, y_pred, labels, "cv")

    @staticmethod
    def __save_results(name: str, y_real: pd.Series, y_pred: pd.Series, labels: [str], option: str) -> None:
        """
        Save results

        :param name: Name of test to save
        :param y_real: Real label
        :param y_pred: Predicted label
        :param labels: Labels name
        :param option: By using cross validation ("cv"), train-test as it is ("split"),
            undersample ("under") or oversample ("over")
        :return: None, generate output
        """
        c_m = confusion_matrix(y_real, y_pred)
        c_m_display = ConfusionMatrixDisplay(c_m)
        c_m_display.plot()
        os.makedirs(os.path.join(RESULTS_FOLDER, name), exist_ok=True)
        folder = os.path.join(RESULTS_FOLDER, name, TYPE_OF_TEST[option])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, "confusion_matrix.png"))
        plt.clf()
        plt.close(c_m_display.figure_)
        with open(os.path.join(folder, "scores.csv"), "w", encoding="utf-8") as file:
            file.write(
                "Accuracy,{acc}\n"
                "{label1}_Precision,{pre_1}\n"
                "{label1}_Recall,{rec_1}\n{label1}_F1-score,{f1_1}\n"
                "{label2}_Precision,{pre_2}\n"
                "{label2}_Recall,{rec_2}\n{label2}_F1-score,{f1_2}\n".format(
                    label1=labels[0], acc=accuracy_score(y_real, y_pred),
                    pre_1=precision_score(y_real, y_pred, pos_label=labels[0], zero_division=0.0),
                    rec_1=recall_score(y_real, y_pred, pos_label=labels[0]),
                    f1_1=f1_score(y_real, y_pred, pos_label=labels[0]),
                    label2=labels[1], pre_2=precision_score(y_real, y_pred, pos_label=labels[1], zero_division=0.0),
                    rec_2=recall_score(y_real, y_pred, pos_label=labels[1]),
                    f1_2=f1_score(y_real, y_pred, pos_label=labels[1])))
