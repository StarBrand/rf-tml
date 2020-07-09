"""process_results.py: Script to join and generate aggregation of resulting metrics"""

import os
import json
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PATH, os.pardir, os.pardir, "data")
RESULTS = os.path.join(DATA, "metrics")
META_DATA = os.path.join(DATA, "meta_data")

os.makedirs(META_DATA, exist_ok=True)

data = pd.DataFrame({}, columns=["Clinical Data", "Mutated Genes",  # Data used to train
                                 "Resampling", "Validation Method", "PCA",  # Pre-processing and tested method
                                 "Accuracy", "M0: F1-score", "M0: Recall", "M0: Precision",  # Metric values (M0)
                                 "M1: F1-score", "M1: Recall", "M1: Precision"])  # Metric values (M1)

with open(os.path.join(META_DATA, "name_of_data.json"), "r") as file:
    info_of_data = json.load(file)


def add_results(path: str, the_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds results of indicated path of results (e.g. clinical_data_count)
    and added to data Pandas.DataFrame

    :param path: Path of results
    :param the_data: Data to change
    :return: Data updated
    """
    path_to_work = os.path.join(RESULTS, path)
    new_row = dict()
    if path.split("_")[-1] == "pca":
        new_row["PCA"] = True
        info = info_of_data[path.replace("_pca", "")]
    else:
        new_row["PCA"] = False
        info = info_of_data[path]
    name = info["Name"]
    new_row["Label"] = info["Label"]
    new_row["Clinical Data"] = info["Clinical Data"]
    new_row["Mutated Genes"] = info["Mutated Genes"]

    def add_info_to_row(a_row: dict, name_sub_folder: str, name_sub_sub_folder: str) -> None:
        """
        Adds info of data to row (a_row, a Python dictionary) based on the
        name of sub-folder

        :param a_row: A row to edit (new_row, always)
        :param name_sub_folder: Sub-folder name (Validation)
        :param name_sub_sub_folder: Sub-sub-folder name (Resample)
        :return: None, alter a_row
        """
        a_row["Validation Method"] = name_sub_folder.replace("_", " ")
        if name_sub_sub_folder == "NoResample":
            a_row["Resampling"] = "None"
        else:
            a_row["Resampling"] = name_sub_sub_folder
        return

    for sub_folder in os.listdir(path_to_work):
        for sub_sub_folder in os.listdir(os.path.join(path_to_work, sub_folder)):
            add_info_to_row(new_row, sub_folder, sub_sub_folder)
            scores = pd.read_csv(os.path.join(path_to_work, sub_folder, sub_sub_folder, "scores.csv"),
                                 sep=",", header=None)
            scores.columns = ["Metric", "Value"]
            for _, metric in scores.iterrows():
                new_row[metric["Metric"].replace("_", ": ")] = metric["Value"]
            the_data = the_data.append(pd.Series(data=new_row.copy(), name=name), ignore_index=False)
    return the_data


if __name__ == "__main__":
    for folder in os.listdir(RESULTS):
        data = add_results(folder, data)
    data.to_csv(os.path.join(META_DATA, "metrics.tsv"), sep="\t")
