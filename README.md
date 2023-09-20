# Repository of datasets used for DS experiments

## Usage
1. The purpose of this repository is to host OctaiPipe's datasets and any accompanying scripts used to prepare the data for an ML use case.
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
| [Batteries](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)    | Experiments on Li-Ion aging batteries. Charging and discharging at different temperatures. Records the impedance as the damage criterion. | No             |
| [Bearings](https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip) | Experiments on bearing failures. Physical data. Classification problem. | No |
| [C-MAPSS 2008](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)       | Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation          | No              |
| [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23) | A labeled dataset with malicious and benign IoT network traffic  | Yes |
| [Li-Ion Battery](https://data.mendeley.com/datasets/nsc7hnsg4s/2) | For health status prediction of lithium-ion batteries ([Ma et al. 2022](https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a))  | Yes |

[^*]: Status indication of whether the dataset has been validated for use for OctaiPipe research.

## References
Below are collections of data sources:
* [`nicolasj/industrial_ml_datasets`](https://github.com/nicolasj92/industrial-ml-datasets) - A curated list of datasets, publically available for machine learning research in the area of manufacturing.
* [Bearing run-to-failure dataset of UNSW - Mendeley Data](https://data.mendeley.com/datasets/h4df4mgrfb/3) - The run-to-failure experiments data were collected at the University of New South Wales in 2019-2020, regarding the development of bearing fault severity assessment methods.
* [NASA Prognostic Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) - The Prognostics Data Repository is a collection of data sets that have been donated by universities, agencies, or companies.
* [`microsoft/BatteryML`](https://github.com/microsoft/BatteryML/blob/main/dataprepare.md) - Microsoft open-sourced tool for R&D of ML on battery degradation.
* GECCO Industrial Challenge [2018](https://www.spotseven.de/gecco/gecco-challenge/gecco-challenge-2018/) and [2019](https://www.th-koeln.de/informatik-und-ingenieurwissenschaften/gecco-challenge-2019_63244.php) - Genetic and Evolutionary Computation Conference, industrial challenge dataset for Online Anomaly Detection for Drinking Water Quality.
* [Cybersecurity dataset for IIoT](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/code) - Dataset for intrusion detection in cyber security applications.
