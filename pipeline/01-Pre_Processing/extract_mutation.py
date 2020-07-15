"""extract_mutation.py: Extract mutation matrix and tml from downloaded data"""

import os
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

PATH = os.path.dirname(os.path.abspath(__file__))

RAW_DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "raw_data")
DATA = os.path.join(PATH, os.pardir, os.pardir, "data", "training_data", "mutated_genes")
os.makedirs(DATA, exist_ok=True)

COLUMNS_TO_USE = ["SYMBOL", "Tumor_Sample_Barcode", "Variant_Classification"]

selector = VarianceThreshold(threshold=0.05)

clinical_sample = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_sample.txt"), sep="\t",
                              skiprows=4, index_col="PATIENT_ID", usecols=["PATIENT_ID", "SAMPLE_ID"])
clinical_patient = pd.read_csv(os.path.join(RAW_DATA, "data_clinical_patient.txt"), sep="\t",
                               skiprows=4, index_col="PATIENT_ID", usecols=["PATIENT_ID", "M_STAGE", "N_STAGE"])
index_dict = clinical_sample["SAMPLE_ID"].to_dict()

data = pd.read_csv(os.path.join(RAW_DATA, "data_mutations_extended.txt"), sep="\t",
                   skiprows=0, header=1, low_memory=False, usecols=COLUMNS_TO_USE)


if __name__ == '__main__':
    # M Stage
    for _, patient in clinical_patient.iterrows():
        if patient["M_STAGE"] == "Mx" or patient["M_STAGE"] is None:
            if patient["N_STAGE"] == "N0":
                patient["M_STAGE"] = "M0"
            elif patient["N_STAGE"] == "N1":
                patient["M_STAGE"] = "M1"
            else:
                patient["M_STAGE"] = None
    clinical_patient.replace({"M1a": "M1", "M1b": "M1", "M2a": "M2", "M2b": "M2"}, inplace=True)
    clinical_patient.drop(columns=["N_STAGE"], inplace=True)
    clinical_patient.dropna(inplace=True)
    clinical_patient.rename(index=index_dict, inplace=True)
    clinical_patient.index.name = "SAMPLE_ID"

    # Mutation matrix
    selected = data[data["Variant_Classification"] == "Missense_Mutation"]
    matrix = pd.DataFrame({})
    for patient in clinical_patient.index:
        matrix = pd.concat(
            [matrix,
             pd.DataFrame(
                 selected[selected["Tumor_Sample_Barcode"] == patient].groupby(["SYMBOL"]).size(),
                 columns=[patient]).T]
        )
    matrix.fillna(value=0, inplace=True)
    new_values = selector.fit(matrix.values)
    matrix = matrix[matrix.columns[selector.get_support(indices=True)]]
    matrix = matrix.astype(int)
    matrix.index.name = "SAMPLE_ID"
    pd.concat([
        matrix, clinical_patient
    ], axis=1, join="inner").to_csv(os.path.join(DATA, "mutated_genes.tsv"), sep="\t", header=True, index=True)

    # TML
    pd.concat(
        [pd.DataFrame(matrix.sum(axis=1), columns=["TML"]),
         clinical_patient],
        axis=1,
        join="inner"
    ).to_csv(
        os.path.join(DATA, "tml.tsv"), sep="\t", header=True, index=True
    )
