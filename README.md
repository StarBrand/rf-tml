# How to replicate experiment

![workflow](https://github.com/StarBrand/rf-tml/wiki/Workflow.png)

## Requirements

The code can be executed on Python`>=3.7`, due to type specification, some code could raise exceptions on Python`<=3.6`.

As a good practice, generating a virtual environment is recommended ([more info](https://docs.python.org/3/tutorial/venv.html)). A quick way to do it:

### By using `conda`

````bash
conda create -n <name_of_venv> python=3.7 # or any python>=3.7

# Activate
conda activate <name_of_venv>
````

### By using Python

*Note*: Need version of Python of virtual environment to create installed

````bash
python -m venv <name_of_venv>
# In case you have installed Python v2
python3 -m venv <name_of_venv>

# Activate
# Windows
<name_of_venv>\Scripts\activate.bat
# Unix or MacOS
source <name_of_venv>/bin/activate
````

### Install libraries

Using pip:

````bash
pip install -r requirements.txt
````

## Preprocessing

![preprocessing](https://github.com/StarBrand/rf-tml/wiki/Pre-Processing.png)

*Note:* `data_mutations_extended.txt`, file that contains the data extracted on [`mutated_genes.tsv`](https://github.com/StarBrand/rf-tml/tree/master/data/training_data/mutated_genes/mutated_genes.tsv) and [`tml.tsv`](https://github.com/StarBrand/rf-tml/tree/master/data/training_data/mutated_genes/tml.tsv), is missing on this repository due to excessive size (0.98 GB). However, it can be downloaded from [cBioPortal](https://www.cbioportal.org/study/summary?id=nsclc_tcga_broad_2016). Files used on this study are `data_mutation_extended.txt`, `data_clinical_patient.txt` and `data_clinical_sample.txt`.

Data (on [`training_data`](https://github.com/StarBrand/rf-tml/tree/master/data/training_data) folder) was generated from data saved on [`raw_data`](https://github.com/StarBrand/rf-tml/tree/master/data/raw_data). In case you want to regenerate, execute:

````bash
cd pipeline/01-Pre_processing
# Generates data/training_data/mutated_genes/*.tsv
python extract_mutation.py
# Generates data/training_data/clinical_data/clinical_data.tsv, not to be used
python clinical_data.py
# Generates data/training_data/clinical_data/encoded_clinical_data.tsv
python one_hot_encoding_clinical_data.py
# Generates merging between clinical data and mutated genes, data/training_data/merged_data/*.tsv
python merge_data.py
cd ../..
````

## Classification

![classification](https://github.com/StarBrand/rf-tml/wiki/Classification.png)

Whole classification is done in one script:

````bash
cd pipeline/02-Classification
python classify.py
# >> A long output
cd ..
````

This generates the whole [`metrics`](https://github.com/StarBrand/rf-tml/tree/master/data/metrics) folder. The output is saved in [`classify.out`](https://github.com/StarBrand/rf-tml/tree/master/data/metrics/classify.out). The specification and parameters are on [`config.py` file](https://github.com/StarBrand/rf-tml/blob/master/classification/models/random_forest/config.py). Seed was fixes arbitrarily to make experiments reproducible.

## Analysis of Result

### Comparing metrics

![metaanalysis](https://github.com/StarBrand/rf-tml/wiki/Meta-Analysis.png)

To compare training data used on model, first, we need to summarize all obtained metrics. In order to do this, execute:

````bash
cd pipeline/03-Meta_Analysis
python process_results.py
# Generates data/meta_data/metrics.tsv
````

Two type of graphs are generating to compare the different data used to train the model. The first one is to compare F1-score of the M1 label (cancer has been found to have spread to distant organs or tissues[*](https://www.cancer.org/treatment/understanding-your-diagnosis/staging.html)). The second one is to compare precision and recall of the same label (M1).

First type of chart is generated with the script:

````bash
python f1_score_plot.py
````

This generate 4 plot:

| **PCA** \| **Model Validation** | 5-Fold Cross Validation | Test Set |
| ------------------------------- | ----------------------- | -------- |
| Without PCA                     | [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/f1-score_cv.png)                | [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/f1-score.png) |
| With PCA                        | [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/f1-score_pca_cv.png)                | [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/f1-score_pca.png) |

Second type of chart is generated by using:

````bash
python precision_recall.py
cd ..
````

This generate 4 plot:

| **Model Validation** | Link to plot |
| ------------------------------- | ------------|
|5-Fold Cross Validation |  [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/precision_recall_cv.png) |
| Test Set | [Plot](https://github.com/StarBrand/rf-tml/blob/master/data/meta_data/precision_recall.png) |

### Getting most important parameters

![featureimportance](https://github.com/StarBrand/rf-tml/wiki/Feature-Importance.png)

To extract feature importance:

````bash
cd 04-Feature-Importance
python extract_feature_importance.py
cd ..
````

This generates `.tsv` files on folder [meta_data](https://github.com/StarBrand/rf-tml/tree/master/data/meta_data).