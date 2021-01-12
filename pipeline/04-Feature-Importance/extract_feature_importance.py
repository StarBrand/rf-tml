import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

CONFIG = {
    "n": 100,
    "jobs": -1,
    "seed": 7,
    "k-Fold": 5,
    "test_size": 0.25
}

PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PATH, os.pardir, os.pardir, "data")
DATA_PATH = os.path.join(DATA, "training_data")
META_DATA = os.path.join(DATA, "meta_data")
LABEL = "M_STAGE"


def import_data(path_to_tsv: str) -> pd.DataFrame:
    return pd.read_csv(path_to_tsv, sep="\t", index_col=0)


# Select data (According to precision_recall(_cv) graph on meta_data)
selected_data = [
    (import_data(os.path.join(DATA_PATH, "clinical_data", "encoded_clinical_data.tsv")), "CF"),
    (import_data(os.path.join(DATA_PATH, "merged_data", "clinical_data_tml.tsv")), "CF + TML"),
    (import_data(os.path.join(DATA_PATH, "clinical_data", "encoded_clinical_data_no_tumor_stage.tsv")), "CF - TS"),
    (import_data(os.path.join(DATA_PATH, "merged_data", "clinical_data_tml_no_stage.tsv")), "CF + TML - TS"),
]


def important_feature(x_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    :param x_train: x_train to fit model
    :param y_train: y_train to fit model
    :return: Dictionary: feature / value
    """
    output = dict()
    random_forest = RandomForestClassifier(n_estimators=CONFIG["n"],
                                           n_jobs=CONFIG["jobs"],
                                           random_state=CONFIG["seed"])
    random_forest.fit(x_train, y_train)
    for label, feature in zip(x_train.columns, random_forest.feature_importances_):
        output[label] = feature
    return output


if __name__ == "__main__":
    sys.path.append(os.path.join(PATH, os.pardir, "02-Classification"))
    from useful_methods import oversample, undersample

    encoding = pd.read_csv(os.path.join(DATA_PATH, "clinical_data", "Encoded_Clinical_Data.txt"), sep="\t")
    for data, name in selected_data:
        print("\tUsing {} to classify".format(name))
        input_data = data.drop(columns=LABEL)
        labels = data[LABEL]
        print("\t\tResample: {}".format("No resample"))
        no_resample = important_feature(input_data, labels)
        print("\t\tResample: {}".format("Undersample"))
        under_sample = important_feature(*undersample(input_data, labels, LABEL))
        print("\t\tResample: {}".format("Oversample"))
        over_sample = important_feature(*oversample(input_data, labels, LABEL))
        feature_importance = pd.DataFrame({
            "No Resample": no_resample,
            "Undersample": under_sample,
            "Oversample": over_sample
        })
        for a_sample in [no_resample, under_sample, over_sample]:
            a_sample["Age"] = a_sample["older"] + a_sample["younger"]
            a_sample.pop("older")
            a_sample.pop("younger")
            for attribute in pd.unique(encoding["Attribute"]):
                a_sample[attribute] = 0
                for cat in encoding[encoding["Attribute"] == attribute]["Encoding name"]:
                    try:
                        a_sample[attribute] += a_sample[cat]
                        a_sample.pop(cat)
                    except KeyError:
                        pass
        feature_importance_decoded = pd.DataFrame({
            "No Resample": no_resample,
            "Undersample": under_sample,
            "Oversample": over_sample
        })
        decoding = encoding.set_index("Encoding name")
        feature_importance.sort_values(by="No Resample", axis=0, ascending=False, inplace=True)
        feature_importance.rename(index=decoding["Category"].to_dict(), inplace=True)
        feature_importance.to_csv(os.path.join(META_DATA, "feature_importance_{}.tsv".format(name)), sep="\t")
        feature_importance_decoded.sort_values(by="No Resample", axis=0, ascending=False, inplace=True)
        feature_importance_decoded.rename(index={"SMOKING_PACK_YEARS": "Smoking Pack Years"}, inplace=True)
        feature_importance_decoded.to_csv(
            os.path.join(META_DATA, "feature_importance_decoded_{}.tsv".format(name)),
            sep="\t"
        )
