# Repository of datasets used for DS experiments

## Usage
1. The purpose of this repository is to host T-DAB datasets and any accompanying scripts used to prepare the data for an ML use case.
2. It should not contain any experimentation notebooks. For this, a separate repo ML-experiments have been created.

## Create Environment
Set up a working miniconda environment as follows:
``` bash
conda create -n datasets python=3.10
conda activate datasets
pip install -r requirements.txt
pre-commit install
```

## Datasets
| Dataset       | Description             | In review?[^*] |
| ------------- |:------------------------|:--------------:|
| [C-MAPSS 2008](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)       | Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation          | No              |
| [Batteries](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)    | Experiments on Li-Ion batteries. Charging and discharging at different temperatures. Records the impedance as the damage criterion. | No             |
| [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23) | A labeled dataset with malicious and benign IoT network traffic  | Yes |

[^*]: Status indication of whether the dataset has been validated for use for T-DAB research.

## References
Below are collections of data sources:
* [`nicolasj/industrial_ml_datasets`](https://github.com/nicolasj92/industrial-ml-datasets) - A curated list of datasets, publically available for machine learning research in the area of manufacturing.
* [Bearing run-to-failure dataset of UNSW - Mendeley Data](https://data.mendeley.com/datasets/h4df4mgrfb/3) - The run-to-failure experiments data were collected at the University of New South Wales in 2019-2020, regarding the development of bearing fault severity assessment methods.
* [NASA Prognostic Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) - The Prognostics Data Repository is a collection of data sets that have been donated by universities, agencies, or companies.
