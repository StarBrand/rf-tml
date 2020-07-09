"""process_results.py: Script to join and generate aggregation of resulting metrics"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

FIGSIZE = (12, 8)
FONTSIZE = 14
TITLESIZE = 16

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

data = pd.read_csv(os.path.join(META_DATA, "metrics.tsv"), header=0, index_col=0, sep="\t")
data_cv = data[data["Validation Method"] == "5-Fold CrossValidation"]
data = data[data["Validation Method"] == "Test Set"]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig_cv, ax_cv = plt.subplots(figsize=FIGSIZE)


def scatter_plot(axes: Axes, df: pd.DataFrame):
    """
    Generates scatter plot to compare precision and recall

    :param axes: Axes to plot
    :param df: DataFrame for data
    :return: None, alter axes
    """
    for resampling, shape in [("None", "o"), ("Oversample", "*"), ("Undersample", "+")]:
        for pca, color in [(True, "b"), (False, "r")]:
            for _, datum in df[(df["Resampling"] == resampling) & (df["PCA"] == pca)].iterrows():
                axes.scatter(datum["M1: Recall"],
                             datum["M1: Precision"],
                             color=color,
                             marker=shape)
                axes.annotate(datum["Label"],
                              (datum["M1: Recall"],
                               datum["M1: Precision"]))
    axes.set_xlabel("Recall", fontsize=FONTSIZE)
    axes.set_ylabel("Precision", fontsize=FONTSIZE)
    return


if __name__ == "__main__":
    scatter_plot(ax, data)
    scatter_plot(ax_cv, data_cv)
    plt.show()
