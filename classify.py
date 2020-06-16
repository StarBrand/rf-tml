import os
import pandas as pd
from parameters import PARAMETERS
from models import RandomForest

data = pd.read_csv(os.path.join("data", "{}.txt".format(PARAMETERS["file"])),
                   sep="\t", index_col=0)

input_data = data.drop(columns=PARAMETERS["columns_to_drop"])
labels = data[PARAMETERS["labels"]]

if __name__ == "__main__":
    random_forest = RandomForest(input_data, labels)
    random_forest.execute_test(
        PARAMETERS["name"],
        PARAMETERS["option"])
