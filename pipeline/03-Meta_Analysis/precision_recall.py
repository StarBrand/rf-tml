"""process_results.py: Script to join and generate aggregation of resulting metrics"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGSIZE = (12, 8)
FONTSIZE = 14
MARKER_SIZE = 200

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

data = pd.read_csv(os.path.join(META_DATA, "metrics.tsv"), header=0, index_col=0, sep="\t")
data_cv = data[data["Validation Method"] == "5-Fold CrossValidation"]
data = data[data["Validation Method"] == "Hold-out"]

fig, ax = plt.subplots(figsize=FIGSIZE)
fig_cv, ax_cv = plt.subplots(figsize=FIGSIZE)


def scatter_plot(axes: Axes, figure: Figure, df: pd.DataFrame) -> None:
    """
    Generates scatter plot to compare precision and recall

    :param axes: Axes to plot
    :param figure: Figure to plot
    :param df: DataFrame for data
    :return: None, alter axes
    """
    legend1 = None
    markers = ["s", "^", "o", "P", "p", "X", "*"]
    labels = sorted(pd.unique(df["Label"]))
    handles = list()
    for index, label in enumerate(labels):
        handles.append(mlines.Line2D([], [], color="#7AC5CD", marker=markers[index], linestyle="None",
                                     markersize=MARKER_SIZE // 10, label=label))
        if len(handles) == 7:
            legend1 = axes.legend(handles=handles, fontsize=FONTSIZE, loc='upper center', bbox_to_anchor=(0.27, -0.1),
                                  ncol=3, fancybox=True, shadow=True)
            handles = list()
        for resampling, color in [("Oversample", "#9ACD32"), ("Undersample", "#C71585"), ("None", "#7AC5CD")]:
            if index == len(labels) - 1:
                handles.append(mlines.Line2D([], [], color=color, marker=markers[0], linestyle="None",
                                             markersize=MARKER_SIZE // 10, label="Resample: {}".format(resampling)))
            for pca in [False, True]:
                if index == len(labels) - 1 and resampling == "None":
                    graphical_params = {
                        "color": "#7AC5CD", "marker": markers[0], "linestyle": "None",
                        "markersize": MARKER_SIZE // 10, "label": "No PCA"
                    }
                    if pca:
                        graphical_params["markeredgecolor"] = "k"
                        graphical_params["markeredgewidth"] = 2.0
                        graphical_params["label"] = "PCA"
                    handles.append(mlines.Line2D([], [], **graphical_params))
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
                    else:
                        graphical_params["edgecolor"] = "gray"
                        graphical_params["linewidth"] = 2.0
                    axes.scatter(datum["M1: Recall"],
                                 datum["M1: Precision"],
                                 **graphical_params)
                except IndexError:
                    pass
    axes.set_xlabel("Recall", fontsize=FONTSIZE)
    axes.set_ylabel("Precision", fontsize=FONTSIZE)
    axes.legend(handles=handles, fontsize=FONTSIZE, loc='upper center', bbox_to_anchor=(0.75, -0.1),
                ncol=2, fancybox=True, shadow=True)
    figure.add_artist(legend1)
    return


if __name__ == "__main__":
    # Hold-out
    scatter_plot(ax, fig, data)
    fig.tight_layout()
    fig.savefig(os.path.join(META_DATA, "precision_recall.png"))

    # Cross Validation
    scatter_plot(ax_cv, fig_cv, data_cv)
    fig_cv.tight_layout()
    fig_cv.savefig(os.path.join(META_DATA, "precision_recall_cv.png"))
