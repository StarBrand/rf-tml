"""plot_feature_importance.py: Plot Obtained feature importance"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PATH = os.path.dirname(os.path.abspath(__file__))
META_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "meta_data")

FIGSIZE = (6, 4)
FONTSIZE = 10
MARKER_SIZE = 25
DELTA = 5.e-3

cf_tml = pd.read_csv(os.path.join(META_DATA, "feature_importance_decoded_CF + TML.tsv"), sep="\t",
                     index_col=0)
cf = pd.read_csv(os.path.join(META_DATA, "feature_importance_decoded_CF.tsv"), sep="\t",
                 index_col=0)


def plot_no_resample(axes: Axes, data: pd.DataFrame) -> None:
    """
    Plots features importance no resample model

    :param axes: An axes to plot
    :param data: Data to plot
    :return: None, alter axes
    """
    axes.barh(data.index, data["No Resample"] - DELTA, height=[0.]*len(data),
              edgecolor="black", linewidth=1.0)
    axes.scatter(data["No Resample"], data.index, marker='o',
    	color="#0080FF", s=MARKER_SIZE, alpha=1.0)
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
    axes.barh(labels - height, under, height=[height]*len(data),
              color="#C71585", label="Undersample", edgecolor="black", linewidth=1.0)
    over = data["Oversample"]
    axes.barh(labels, over, height=[height]*len(data),
              color="#9ACD32", label="Oversample", edgecolor="black", linewidth=1.0)
    test_set = data["No Resample"]
    axes.barh(labels + height, test_set, height=[height]*len(data),
              color="#7AC5CD", label="No resample", edgecolor="black", linewidth=1.0)
    axes.set_yticks(labels)
    axes.set_yticklabels(under.index)
    axes.set_xlabel("Feature Importance", fontsize=FONTSIZE)
    axes.legend(fontsize=FONTSIZE)
    return


if __name__ == "__main__":
    # Clinical Feature
    # No resample
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot_no_resample(ax, cf)
    fig.tight_layout()
    fig.savefig(os.path.join(META_DATA, "feature_importance_CF.png"))

    # Resample
    fig_re, ax_re = plt.subplots(figsize=FIGSIZE)
    plot_resample(ax_re, cf)
    fig_re.tight_layout()
    fig_re.savefig(os.path.join(META_DATA, "feature_importance_CF_resample.png"))

    # Clinical Feature + TML
    # No resample
    fig_tml, ax_tml = plt.subplots(figsize=FIGSIZE)
    plot_no_resample(ax_tml, cf_tml)
    fig_tml.tight_layout()
    fig_tml.savefig(os.path.join(META_DATA, "feature_importance_CF+TML.png"))

    # Resample
    fig_tml_re, ax_tml_re = plt.subplots(figsize=FIGSIZE)
    plot_resample(ax_tml_re, cf)
    fig_tml_re.tight_layout()
    fig_tml_re.savefig(os.path.join(META_DATA, "feature_importance_CF+TML_resample.png"))
