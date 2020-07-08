"""f1_score_plot.py: Generates plot of F1-score of results"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

FIGSIZE = (12, 8)
FONTSIZE = 14
TITLESIZE = 16
DEGREES = 80

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, "data", "meta_data")

data = pd.read_csv(os.path.join(META_DATA, "metrics.csv"), header=0, index_col=0, sep="\t")
data_pca = data[data["PCA"]]
data = data[data["PCA"] == False]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig_pca, ax_pca = plt.subplots(figsize=FIGSIZE)


def bar_plot(axes: Axes, df: pd.DataFrame, title: str) -> None:
    """
    Does bar plot of df (pca and with no pca)

    :param axes: Axes to plot data
    :param df: DataFrame to put on data
    :param title: Title of graph
    :return: None, alter axes
    """
    width = 1.0 / 4.5
    cross_validation = df[df["Cross Validation"] != "None"]
    labels = np.arange(len(cross_validation))
    axes.bar(labels - (3 * width / 2), cross_validation["M1: F1-score"], width, color="#7AC5CD", label="5-Fold Cross Validation")
    under = df[df["Resampling"] == "Undersample"]
    axes.bar(labels - (width / 2), under["M1: F1-score"], width, color="#C71585", label="Undersample")
    over = df[df["Resampling"] == "Oversample"]
    axes.bar(labels + (width / 2) , over["M1: F1-score"], width, color="#9ACD32", label="Oversample")
    test_set = df[(df["Resampling"] == "None") & (df["Cross Validation"] == "None")]
    axes.bar(labels + (3 * width / 2), test_set["M1: F1-score"], width, color="#FFA500", label="Test Set 25%")
    axes.legend(fontsize=FONTSIZE)
    axes.set_xticks(labels)
    axes.tick_params(axis='x', rotation=DEGREES)
    axes.set_xticklabels([tick_label.replace(", ", ",\n").replace(" and ", "\nand ") for tick_label in cross_validation.index])
    axes.set_ylabel("F1-score, M1 prediction")
    axes.set_title("{}\n".format(title), fontsize=TITLESIZE)
    return 


if __name__ == "__main__":
    # No PCA
    bar_plot(ax, data, "F1-score of different input data")
    fig.tight_layout()
    fig.savefig(os.path.join(META_DATA, "f1-score.png"))

    # PCA
    bar_plot(ax_pca, data_pca, "F1-score of different input data with PCA")
    fig_pca.tight_layout()
    fig_pca.savefig(os.path.join(META_DATA, "f1-score-pca.png"))
