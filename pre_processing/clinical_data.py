"""clinical_data.py: Select clinical data to use on classification"""

import os
import pandas as pd

RAW_DATA = os.path.join(os.pardir, "raw_data")
DATA = os.path.join(os.pardir, "data")

SELECTED_COLS = [
    "PATIENT_ID", "AGE", "SEX",
    "SMOKING_HISTORY", "SMOKING_PACK_YEARS",
    "STAGE"
]

data_patient = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_patient.txt"), sep="\t",
                           index_col="PATIENT_ID", skiprows=4, usecols=SELECTED_COLS)
data_sample = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_sample.txt"), sep="\t",
                          index_col="PATIENT_ID", skiprows=4, usecols=["PATIENT_ID", "SAMPLE_ID"])
m_stage = pd.read_csv(os.path.join(DATA, "mutated_genes", "mutated_genes.csv"), sep="\t",
                      index_col="SAMPLE_ID", usecols=["SAMPLE_ID", "M_STAGE"])
data = pd.concat([data_patient, data_sample], axis=1, join="inner")
data.set_index("SAMPLE_ID", inplace=True)
data = pd.concat([data, m_stage], axis=1, join="inner")

if __name__ == '__main__':
    data.to_csv(os.path.join(DATA, "clinical_data", "clinical_data.csv"), sep="\t")
