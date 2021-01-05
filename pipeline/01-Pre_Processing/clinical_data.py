"""clinical_data.py: Select clinical data to use on classification"""

import os
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

RAW_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "raw_data")
DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "training_data")

SELECTED_COLS = [
    "PATIENT_ID", "AGE",
    "SMOKING_HISTORY", "STAGE"
]

data_patient = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_patient.txt"), sep="\t",
                           index_col="PATIENT_ID", skiprows=4, usecols=SELECTED_COLS)

mask = data_patient["SMOKING_HISTORY"].str.contains("Current Reformed Smoker")
mask.fillna(False, inplace=True)
data_patient.loc[mask, "SMOKING_HISTORY"] = "Current Reformed Smoker"

data_sample = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_sample.txt"), sep="\t",
                          index_col="PATIENT_ID", skiprows=4,
                          usecols=["PATIENT_ID", "SAMPLE_ID", "CANCER_TYPE_DETAILED"])
m_stage = pd.read_csv(os.path.join(DATA, "mutated_genes", "mutated_genes.tsv"), sep="\t",
                      index_col="SAMPLE_ID", usecols=["SAMPLE_ID", "M_STAGE"])
data = pd.concat([data_patient, data_sample], axis=1, join="inner")
data.set_index("SAMPLE_ID", inplace=True)
data = pd.concat([data, m_stage], axis=1, join="inner")


if __name__ == '__main__':
    # "Raw" clinical data
    data.to_csv(os.path.join(DATA, "clinical_data", "clinical_data.tsv"), sep="\t")

    # No tumor stage clinical data
    data.drop(columns=["STAGE"], inplace=True)
    data.to_csv(os.path.join(DATA, "clinical_data", "clinical_data_no_tumor_stage.tsv"), sep="\t")
