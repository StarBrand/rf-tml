"""plot_feature_importance.py: Plot Obtained feature importance"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

FIGSIZE = (6, 4)
FONTSIZE = 12
MARKER_SIZE = 25
DELTA = 5.e-3


def import_feature_data(particular_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(META_DATA, particular_path), sep="\t", index_col=0)


data_list = [
    (import_feature_data("feature_importance_decoded_CF + TML.tsv").iloc[0:5].iloc[::-1], "CF+TML"),
    (import_feature_data("feature_importance_decoded_CF.tsv").iloc[0:5].iloc[::-1], "CF"),
    (import_feature_data("feature_importance_CF + TML.tsv").iloc[0:5].iloc[::-1], "CF+TML_undecoded"),
    (import_feature_data("feature_importance_CF.tsv").iloc[0:5].iloc[::-1], "CF_undecoded"),
    (import_feature_data("feature_importance_decoded_CF - TS.tsv").iloc[0:5].iloc[::-1], "CF-TS"),
    (import_feature_data("feature_importance_decoded_CF + TML - TS.tsv").iloc[0:5].iloc[::-1], "CF+TML-TS"),
    (import_feature_data("feature_importance_CF - TS.tsv").iloc[0:5].iloc[::-1], "CF-TS_undecoded"),
    (import_feature_data("feature_importance_CF + TML - TS.tsv").iloc[0:5].iloc[::-1], "CF+TML-TS_undecoded")
]


def plot_no_resample(axes: Axes, data: pd.DataFrame) -> None:
    """
    Plots features importance no resample model

    :param axes: An axes to plot
    :param data: Data to plot
    :return: None, alter axes
    """
    axes.barh(data.index, data["No Resample"] - DELTA, height=[0.] * len(data),
              edgecolor="black", linewidth=1.0)
    axes.scatter(data["No Resample"], data.index, marker='o',
                 color="#0080FF", s=MARKER_SIZE, alpha=1.0)
    axes.set_yticklabels(data.index, fontsize=FONTSIZE)
    axes.set_xlabel("Feature Importance", fontsize=FONTSIZE)
    return


def plot_resample(axes: Axes, data: pd.DataFrame) -> None:
    """
    Plots features importance resample model

    :param axes: An axes to plot
    :param data: Data to plot
    :return: None, alter axes
    """
    height = 1.0 / 3.5
    under = data["Undersample"]
    labels = np.arange(len(under))
    axes.barh(labels - height, under, height=[height] * len(data),
              color="#C71585", label="Undersample", edgecolor="black", linewidth=1.0)
    over = data["Oversample"]
    axes.barh(labels, over, height=[height] * len(data),
              color="#9ACD32", label="Oversample", edgecolor="black", linewidth=1.0)
    test_set = data["No Resample"]
    axes.barh(labels + height, test_set, height=[height] * len(data),
              color="#7AC5CD", label="No resample", edgecolor="black", linewidth=1.0)
    axes.set_yticks(labels)
    axes.set_yticklabels(under.index, fontsize=FONTSIZE)
    axes.set_xlabel("Feature Importance")
    axes.legend(fontsize=FONTSIZE)
    return


if __name__ == "__main__":
    for a_data, name in data_list:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        plot_no_resample(ax, a_data)
        fig.tight_layout()
        fig.savefig(os.path.join(META_DATA, "feature_importance_{}.png".format(name)))
        fig_re, ax_re = plt.subplots(figsize=FIGSIZE)
        plot_resample(ax_re, a_data)
        fig_re.tight_layout()
        fig_re.savefig(os.path.join(META_DATA, "feature_importance_{}_resample.png".format(name)))
