import os
import pandas as pd


not_norm_genes = pd.read_csv(os.path.join(os.pardir, "data", "countMutMatrix_filtered_ZeroVar_nonNorm.txt"), sep="\t")
not_norm_genes.rename(columns={"PATIENT_ID": "SAMPLE_ID"}, inplace=True)
not_norm_genes.set_index("SAMPLE_ID", inplace=True)
m_stage = pd.read_csv(os.path.join(os.pardir, "data", "input_random_forest_genes.txt"), sep="\t",
                      usecols=["SAMPLE_ID", "M_STAGE"])
m_stage.set_index("SAMPLE_ID", inplace=True)
data_processed = pd.concat([not_norm_genes, m_stage], axis=1, join="inner")

if __name__ == '__main__':
    data_processed.to_csv(os.path.join(os.pardir, "data", "input_rf_genes_nonNorm.txt"), sep="\t")
