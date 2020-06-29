# Random Forest to predict Metastasis Stage
Classification of mutation and data clinical to predict metastasis stage

## How to replicate experiment

### Requirements

The code can be executed on Python`>=3.7`, due to type specification, some code could raise exceptions on Python`<=3.6`.

As a good practice, generating a virtual environment is recommended ([more info](https://docs.python.org/3/tutorial/venv.html)). A quick way to do it:

#### By using `conda`

````bash
conda create -n <name_of_venv> python=3.7 # or any python>=3.7

# Activate
conda activate <name_of_venv>
````

#### By using Python

*Note*: Need version of Python of virtual environment to create installed

````bash
python -m venv <name_of_venv>
# In case you have installed Python v2
python3 -m venv <name_of_venv>

# Activate
## Windows
<name_of_venv>\Scripts\activate.bat
## Unix or MacOS
source <name_of_venv>/bin/activate
````

#### Install libraries

Using pip:

````bash
pip install -r requirements.txt
````

### Preprocessing

Data (on [`data`](https://github.com/StarBrand/rf-tml/tree/master/data) folder) was generated from data saved on [`raw_data`](https://github.com/StarBrand/rf-tml/tree/master/raw_data). In case want to regenerate, execute:

````bash
cd pre_processing
# Generates data/clinical_data/clinical_data.csv, not to be used
python clinical_data.py
# Generates data/clinical_data/encoded_clinical_data.csv
python one_hot_encoding_clinical_data.py
# Generates merging between clinical data and mutated genes
python merge_data.py
cd ..
````

### Classification

Whole classification is done in one script:

````bash
cd classification
python classify.py
# >> A long output
cd ..
````

This generates the whole [`results`](https://github.com/StarBrand/rf-tml/tree/master/results) folder and the output is storage in [`classify.out`](https://github.com/StarBrand/rf-tml/tree/master/classify.out). The specification and parameters are on [`config.py` file](https://github.com/StarBrand/rf-tml/blob/master/classification/models/random_forest/config.py). Seed was fixes arbitrarily to make experiments reproducible.