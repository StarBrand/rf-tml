import os
import pandas as pd


data = pd.read_csv(os.path.join(os.pardir, "data", "input_random_forest_genes.txt"), sep="\t",
                   index_col=0)
sub_data = pd.concat([data["M_STAGE"], data["countMut"]], axis=1)

if __name__ == '__main__':
    sub_data.to_csv(os.path.join(os.pardir, "data", "input_rf_genes_count_mut.txt"), sep="\t")
