# Repository of Datasets and Usecases for DS Benchmarking Experiments

## Usage
1. The purpose of this repository is to host OctaiPipe's datasets and any accompanying scripts used to prepare the data for an ML use case.
2. It should not contain any experimentation notebooks. For this, a separate repo ML-experiments have been created.

## Create Environment
Set up a working miniconda environment as follows:
``` bash
conda create -n datasets python=3.9
conda activate datasets
pip install -r requirements.txt
pre-commit install
```

## Datasets
| Topics | Dataset       | Description             | Use case |
|:------:| ------------- |:------------------------|:--------:|
|PHM| [Batteries-NASA](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip)    | Experiments on Li-Ion aging batteries. Charging and discharging at different temperatures. Records the impedance as the damage criterion. ([Saha et al. 2009](https://ieeexplore.ieee.org/abstract/document/4655607)) | Battery health monitoring |
|PHM| [Batteries-NNSF](https://data.mendeley.com/datasets/nsc7hnsg4s/2) | For health status prediction of lithium-ion batteries ([Ma et al. 2022](https://pubs.rsc.org/en/content/articlehtml/2022/ee/d2ee01676a))  | Battery health monitoring |
|PHM| [Bearings-NASA](https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip) | Rolling element bearing prognostics using physical experiments. ([Qiu et al. 2006](https://doi.org/10.1016/j.jsv.2005.03.007)) | Bearing health status prediction |
|PHM| [C-MAPSS 2008-NASA](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)       | Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. ([Saxena et al. 2008](https://ieeexplore.ieee.org/document/4711414)) | RUL prediction |
|Cybersecurity| [N-BaIoT](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset)[^*] | N-BaIoT: Dataset to Detect IoT Botnet Attacks ([Meidan et al. 2018](https://arxiv.org/abs/1805.03409)) | 1-threat detection (botnet)
|Cybersecurity| [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23) | A labeled dataset with malicious and benign IoT network traffic ([Garcia et al. 2020](http://doi.org/10.5281/zenodo.4743746), Avast)  | 1-threat detection (malware) |
|Cybersecurity| [Edge-IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/code) | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning ([Ferrag et al. 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9751703)) | 5-threats detection |
| Environment | GECCO [2018](https://www.spotseven.de/gecco/gecco-challenge/gecco-challenge-2018/), [2019](https://www.th-koeln.de/informatik-und-ingenieurwissenschaften/gecco-challenge-2019_63244.php) | Internet of Things: Online Anomaly Detection for Drinking Water Quality (Rehbach et al. 2018, 2019) | Water quality monitoring |


[^*]: Seminal paper

## References
Below are collections of data sources:
* [`nicolasj/industrial_ml_datasets`](https://github.com/nicolasj92/industrial-ml-datasets) - A curated list of datasets, publically available for machine learning research in the area of manufacturing.
* [Bearing run-to-failure dataset of UNSW - Mendeley Data](https://data.mendeley.com/datasets/h4df4mgrfb/3) - The run-to-failure experiments data were collected at the University of New South Wales in 2019-2020, regarding the development of bearing fault severity assessment methods.
* [NASA Prognostic Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) - The Prognostics Data Repository is a collection of data sets that have been donated by universities, agencies, or companies.
* [`microsoft/BatteryML`](https://github.com/microsoft/BatteryML/blob/main/dataprepare.md) - Microsoft open-sourced tool for R&D of ML on battery degradation.
