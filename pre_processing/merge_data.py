"""merge_data.py: Merge clinical data and mutates genes"""

import os
import pandas as pd

DATA = os.path.join(os.pardir, "data")

clinical_data = pd.read_csv(os.path.join(DATA, "clinical_data", "encoded_clinical_data.csv"),
                            sep="\t", index_col="SAMPLE_ID")
clinical_data.drop(columns="M_STAGE", inplace=True)
mutated_genes = pd.read_csv(os.path.join(DATA, "mutated_genes", "mutated_genes.csv"),
                            sep="\t", index_col="SAMPLE_ID")
count_mut = pd.read_csv(os.path.join(DATA, "mutated_genes", "count_mut.csv"),
                        sep="\t", index_col="SAMPLE_ID")
mutated_genes_not_norm = pd.read_csv(os.path.join(DATA, "mutated_genes", "mutated_genes_not_norm.csv"),
                                     sep="\t", index_col="SAMPLE_ID")
count_mut_not_norm = pd.read_csv(os.path.join(DATA, "mutated_genes", "count_mut_not_norm.csv"),
                                 sep="\t", index_col="SAMPLE_ID")

if __name__ == '__main__':
    data_folder = os.path.join(DATA, "merged_data")
    os.makedirs(data_folder, exist_ok=True)

    # Norm
    data = pd.concat([clinical_data, mutated_genes], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_mut.csv"), sep="\t")
    data.drop(columns="M_STAGE", inplace=True)
    data = pd.concat([data, count_mut], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_mut_count.csv"), sep="\t")
    data = pd.concat([clinical_data, count_mut], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_count.csv"), sep="\t")

    # Not Norm
    data = pd.concat([clinical_data, mutated_genes_not_norm], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_mut_not_norm.csv"), sep="\t")
    data.drop(columns="M_STAGE", inplace=True)
    data = pd.concat([data, count_mut_not_norm], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_mut_count_not_norm.csv"), sep="\t")
    data = pd.concat([clinical_data, count_mut_not_norm], axis=1, join="inner")
    data.to_csv(os.path.join(data_folder, "clinical_data_count_not_norm.csv"), sep="\t")
