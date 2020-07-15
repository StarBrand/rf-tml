"""merge_data.py: Merge clinical data and mutates genes"""

import os
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "training_data")

clinical_data = pd.read_csv(os.path.join(DATA, "clinical_data", "encoded_clinical_data.tsv"),
                            sep="\t", index_col="SAMPLE_ID")
clinical_data.drop(columns="M_STAGE", inplace=True)
clinical_data_no_stage = pd.read_csv(os.path.join(DATA, "clinical_data", "encoded_clinical_data_no_tumor_stage.tsv"),
                                     sep="\t", index_col="SAMPLE_ID")
clinical_data_no_stage.drop(columns="M_STAGE", inplace=True)
mutated_genes = pd.read_csv(os.path.join(DATA, "mutated_genes", "mutated_genes.tsv"),
                            sep="\t", index_col="SAMPLE_ID")
tml = pd.read_csv(os.path.join(DATA, "mutated_genes", "tml.tsv"),
                  sep="\t", index_col="SAMPLE_ID")


if __name__ == '__main__':
    data_folder = os.path.join(DATA, "merged_data")
    os.makedirs(data_folder, exist_ok=True)

    # Norm
    data = pd.concat([clinical_data, mutated_genes], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_mut.tsv"), sep="\t")
    data = pd.concat([clinical_data, tml], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_tml.tsv"), sep="\t")
    data = pd.concat([clinical_data_no_stage, tml], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_tml_no_stage.tsv"), sep="\t")
