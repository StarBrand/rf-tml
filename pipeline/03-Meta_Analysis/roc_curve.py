"""roc_curve.py: Script to generate ROC curve"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

FIGSIZE = (12, 8)
FONTSIZE = 14
LINEWIDTH = 2.5
COLORS = ["#7AC5CD", "#9ACD32", "#C71585"]

PATH = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PATH, os.pardir, os.pardir, "data")
TRAINING_DATA_PATH = os.path.join(DATA_PATH, "training_data")

TRAINING_DATA = [(os.path.join(TRAINING_DATA_PATH, "merged_data", "clinical_data_tml.tsv"), "CF + TML"),
                 (os.path.join(TRAINING_DATA_PATH, "clinical_data", "encoded_clinical_data.tsv"), "CF"),
                 (os.path.join(TRAINING_DATA_PATH, "mutated_genes", "tml.tsv"), "TML")]

TRAINING_DATA_NO_TS = [(os.path.join(TRAINING_DATA_PATH, "merged_data", "clinical_data_tml_no_stage.tsv"),
                        "CF + TML - TS"),
                       (os.path.join(TRAINING_DATA_PATH, "clinical_data", "encoded_clinical_data_no_tumor_stage.tsv"),
                        "CF - TS"),
                       (os.path.join(TRAINING_DATA_PATH, "mutated_genes", "tml.tsv"), "TML")]


def generate_data_to_use(path_to_tsv: str) -> (pd.DataFrame, pd.Series):
    """
    Imports data and separates it in a DataFrame and Series for labels

    :param path_to_tsv: Path to the file with data
    :return: A tuple of data and label
    """
    imported_data = pd.read_csv(path_to_tsv, sep="\t", index_col=0)
    return imported_data.drop(columns="M_STAGE"), imported_data["M_STAGE"]


def plot_roc_curve(path_to_tsv: str, name: str, axes: Axes, color: str) -> None:
    """
    Plot a roc curve for a particular input data on model

    :param path_to_tsv: Path to the file with data
    :param name: Name of data model
    :param axes: An Axes object
    :param color: Color of line
    :return: None, alter axes
    """
    data, labels = generate_data_to_use(path_to_tsv)
    label_names = list(sorted(pd.unique(labels)))
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=CONFIG["test_size"],
                                                        random_state=CONFIG["seed"])
    model = RandomForest(name)
    y_test, y_pred = model.train_test(x_train, x_test, y_train, y_test, "none")
    y_test = np.equal(y_test, label_names[1])
    y_pred = np.equal(y_pred, label_names[1])
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    axes.plot(fpr, tpr, c=color, linewidth=LINEWIDTH, label=name)
    return


def set_plot(axes: Axes) -> None:
    """
    Set plot to export

    :param axes: Axes object
    :return: None, alter axes
    """
    axes.set_xlabel("False Positive Rate", fontsize=FONTSIZE)
    axes.set_ylabel("True Positive Rate", fontsize=FONTSIZE)
    axes.set_xlim(-0.01, 1.01)
    axes.set_ylim(-0.01, 1.01)
    axes.legend(fontsize=FONTSIZE)


if __name__ == "__main__":
    sys.path.append(os.path.join(PATH, os.pardir, "02-Classification"))
    from models import CONFIG, RandomForest

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], c='k')
    colors = COLORS.copy()

    for a_path, a_name in TRAINING_DATA:
        plot_roc_curve(a_path, a_name, ax, colors.pop())

    set_plot(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_PATH, "meta_data", "roc_curve.png"))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], c='k')
    colors = COLORS.copy()

    for a_path, a_name in TRAINING_DATA_NO_TS:
        plot_roc_curve(a_path, a_name, ax, colors.pop())

    set_plot(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_PATH, "meta_data", "roc_curve_no_ts.png"))
