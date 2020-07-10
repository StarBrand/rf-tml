"""process_results.py: Script to join and generate aggregation of resulting metrics"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

FIGSIZE = (12, 8)
FONTSIZE = 14
TITLESIZE = 16
MARKER_SIZE = 200

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

data = pd.read_csv(os.path.join(META_DATA, "metrics.tsv"), header=0, index_col=0, sep="\t")
data_cv = data[data["Validation Method"] == "5-Fold CrossValidation"]
data = data[data["Validation Method"] == "Test Set"]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig_cv, ax_cv = plt.subplots(figsize=FIGSIZE)


def scatter_plot(axes: Axes, df: pd.DataFrame, title: str):
    """
    Generates scatter plot to compare precision and recall

    :param axes: Axes to plot
    :param df: DataFrame for data
    :param title: Title of plot
    :return: None, alter axes
    """
    datum_labeled = list()
    first_color = 3
    already_pca = False
    for resampling, shape in [("None", "s"), ("Oversample", "^"), ("Undersample", "o")]:
        first_shape = True
        for pca, color in [(True, "k"), (False, "w")]:
            for _, datum in df[(df["Resampling"] == resampling) & (df["PCA"] == pca)].iterrows():
                graphical_params = {
                    "edgecolor": 'k',
                    "facecolor": color,
                    "marker": shape,
                    "s": MARKER_SIZE
                }
                if first_shape:
                    graphical_params["label"] = "Resample: {}".format(resampling)
                    first_shape = False
                    first_color -= 1
                elif first_color == 0:
                    if pca and not already_pca:
                        graphical_params["label"] = "PCA"
                        already_pca = True
                    elif not pca:
                        graphical_params["label"] = "No PCA"
                        first_color -= 1
                axes.scatter(datum["M1: Recall"],
                             datum["M1: Precision"],
                             **graphical_params)
                axes.annotate(datum["Label"],
                              (datum["M1: Recall"] - 0.02,
                               datum["M1: Precision"] - 0.05))
    axes.set_xlabel("Recall", fontsize=FONTSIZE)
    axes.set_ylabel("Precision", fontsize=FONTSIZE)
    axes.set_title("{}\n".format(title), fontsize=TITLESIZE)
    axes.legend(fontsize=FONTSIZE, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                ncol=2, fancybox=True, shadow=True)
    return


if __name__ == "__main__":
    # Test Set
    scatter_plot(ax, data, "Validation Method: Test Set")
    fig.tight_layout()
    fig.savefig(os.path.join(META_DATA, "precision_recall.png"))

    # Cross Validation
    scatter_plot(ax_cv, data_cv, "Validation Method: 5-Fold Cross Validation")
    fig_cv.tight_layout()
    fig_cv.savefig(os.path.join(META_DATA, "precision_recall_cv.png"))
