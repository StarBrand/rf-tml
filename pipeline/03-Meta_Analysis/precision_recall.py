"""process_results.py: Script to join and generate aggregation of resulting metrics"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    markers = ["s", "^", "o", "P", "p", "X", "*"]
    labels = sorted(pd.unique(df["Label"]))
    handles = list()
    for index, label in enumerate(labels):
        handles.append(mlines.Line2D([], [], color="#7AC5CD", marker=markers[index], linestyle="None",
                                     markersize=MARKER_SIZE // 10, label=label))
        for pca in [False, True]:
            if index == len(labels) - 1:
                graphical_params = {
                    "color": "#7AC5CD", "marker": markers[0], "linestyle": "None",
                    "markersize": MARKER_SIZE // 10, "label": "No PCA"
                }
                if pca:
                    graphical_params["markeredgecolor"] = "k"
                    graphical_params["markeredgewidth"] = 2.0
                    graphical_params["label"] = "PCA"
                handles.append(mlines.Line2D([], [], **graphical_params))
            for resampling, color in [("None", "#7AC5CD"), ("Oversample", "#9ACD32"), ("Undersample", "#C71585")]:
                if index == len(labels) - 1 and pca:
                    handles.append(mlines.Line2D([], [], color=color, marker=markers[0], linestyle="None",
                                                 markersize=MARKER_SIZE // 10, label="Resample: {}".format(resampling)))
                try:
                    datum = df[
                        (df["PCA"] == pca) &
                        (df["Resampling"] == resampling) &
                        (df["Label"] == label)
                    ].iloc[0]
                    graphical_params = {
                        "facecolor": color,
                        "marker": markers[index],
                        "s": MARKER_SIZE
                    }
                    if pca:
                        graphical_params["edgecolor"] = "k"
                        graphical_params["linewidth"] = 2.0
                    line = axes.scatter(datum["M1: Recall"],
                                        datum["M1: Precision"],
                                        **graphical_params)
                except IndexError:
                    pass
    axes.set_xlabel("Recall", fontsize=FONTSIZE)
    axes.set_ylabel("Precision", fontsize=FONTSIZE)
    axes.set_title("{}\n".format(title), fontsize=TITLESIZE)
    axes.legend(handles=handles, fontsize=FONTSIZE, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                ncol=4, fancybox=True, shadow=True)
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
