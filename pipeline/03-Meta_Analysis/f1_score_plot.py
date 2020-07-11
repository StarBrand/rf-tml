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
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

data = pd.read_csv(os.path.join(META_DATA, "metrics.tsv"), header=0, index_col=0, sep="\t")
data_pca = data[data["PCA"]]
data = data[~ data["PCA"]]
to_plot = {
    "pca": {
        "cv": data_pca[data_pca["Validation Method"] == "5-Fold CrossValidation"],
        "split": data_pca[data_pca["Validation Method"] == "Test Set"]
    },
    "no_pca": {
        "cv": data[data["Validation Method"] == "5-Fold CrossValidation"],
        "split": data[data["Validation Method"] == "Test Set"]
    }
}

fig, ax = plt.subplots(figsize=FIGSIZE)
fig_cv, ax_cv = plt.subplots(figsize=FIGSIZE)
fig_pca, ax_pca = plt.subplots(figsize=FIGSIZE)
fig_pca_cv, ax_pca_cv = plt.subplots(figsize=FIGSIZE)


def bar_plot(axes: Axes, df: pd.DataFrame, title: str) -> None:
    """
    Does bar plot of df (pca and with no pca)

    :param axes: Axes to plot data
    :param df: DataFrame to put on data
    :param title: Title of graph
    :return: None, alter axes
    """
    width = 1.0 / 3.5
    under = df[df["Resampling"] == "Undersample"]
    labels = np.arange(len(under))
    axes.bar(labels - width, under["M1: F1-score"], width,
             color="#C71585", label="Undersample", edgecolor="black", linewidth=1.0)
    over = df[df["Resampling"] == "Oversample"]
    axes.bar(labels, over["M1: F1-score"], width,
             color="#9ACD32", label="Oversample", edgecolor="black", linewidth=1.0)
    test_set = df[df["Resampling"] == "None"]
    axes.bar(labels + width, test_set["M1: F1-score"], width,
             color="#7AC5CD", label="No resample", edgecolor="black", linewidth=1.0)
    axes.legend(fontsize=FONTSIZE)
    axes.set_xticks(labels)
    axes.tick_params(axis='x', rotation=DEGREES, labelsize=FONTSIZE)
    axes.set_xticklabels([tick_label.replace(" +", "\n+").replace(" -", "\n-") for tick_label in under.index])
    axes.set_ylabel("F1-score, M1 prediction", fontsize=FONTSIZE)
    axes.set_title("{}\n".format(title), fontsize=TITLESIZE)
    return


if __name__ == "__main__":
    # No PCA
    # Test Set
    bar_plot(ax, to_plot["no_pca"]["split"], "F1-score of different input data")
    fig.tight_layout()
    fig.savefig(os.path.join(META_DATA, "f1-score.png"))

    # Cross Validation
    bar_plot(ax_cv, to_plot["no_pca"]["cv"], "F1-score of different input data")
    fig_cv.tight_layout()
    fig_cv.savefig(os.path.join(META_DATA, "f1-score_cv.png"))

    # PCA
    # Test Set
    bar_plot(ax_pca, to_plot["pca"]["split"], "F1-score of different input data")
    fig_pca.tight_layout()
    fig_pca.savefig(os.path.join(META_DATA, "f1-score_pca.png"))

    # Cross Validation
    bar_plot(ax_pca_cv, to_plot["pca"]["cv"], "F1-score of different input data")
    fig_pca_cv.tight_layout()
    fig_pca_cv.savefig(os.path.join(META_DATA, "f1-score_pca_cv.png"))
