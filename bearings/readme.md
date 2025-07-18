# The NASA bearing dataset
The [NASA bearing dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) is one provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati. See also [Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset), collected from experiments in which 4 bearings installed on a test rig were run to failure. Accelerometer readings from each bearing were collected. There were data from three experiments available. Labels are available for the first experiment ('first dataset'), although the original source of such labels is no longer traceable. 'At the end of the test-to-failure experiment, inner race defect occurred in bearing 3 and roller element defect in bearing 4.'

The processed bearing data files are stored on our `octaipipedatasets` blob storage, in the `nasa-bearings` container.



References:
- https://github.com/Miltos-90/Failure_Classification_of_Bearings
- https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
- https://www.kaggle.com/code/furkancitil/nasa-bearing-dataset-supervised-learning


## Information about the raw dataset
Following these references, we only consider the first dataset of the NASA bearing dataset. Each file in the first-dataset directory contains the data at a particular timestamp (given by file name). Such data is that of the 2 channels (accelerometer readings) of 4 bearings-- 8 columns in total, and is tab-separated. The data is 1-second worth of vibration signal recorded at that timestamp; each file consists of 20,480 points with the sampling rate set at 20 kHz.

The bearings each has the following classes:

```
Bearing 1:
"early", "suspect", "normal", "imminent_failure"
Bearing 2:
"early", "suspect", "normal", "imminent_failure"
Bearing 3:
"early", "suspect", "normal", "Inner_race_failure"
Bearing 4:
"early", "suspect", "normal", "Inner_race_failure", "Stage_two_failure"
```

## Data preparation

The bearing data files in this dataset repo and on our OctaiPipe Dataset blob storage are results of a data preparation pipeline, based on our previous ML-experiment, as was done in https://github.com/OctaiPipe/ml-experiments/blob/39e87f7f3c15e26e54620885c6144e44e238fa43/src/federated_learning/bearings_pytorch_convnet/nasa-bearing-dataset-classification-problem.ipynb.

In the following we describe the steps involved.

### 0. Download the dataset

Download the dataset from either of:
- https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
- https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository Section 4
- https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

### 1. Data ingestion and feature engineering

- Data ingestion and feature engineering are done with the use of the np.genfromtxt function
- From each text file containing 1-second worth of vibration signal recorded at that timestamp--- each file consists of 20,480 points with the sampling rate set at 20 kHz, on each sensor channel of each bearing, we compute the following time-domain and frequency-domain features:

    - Time-domain features

```
time_features_map = {'rms': compute_rms,
                     'max_abs': compute_max_abs,
                     'entropy': compute_shannon_entropy,
                     'p2p': compute_p2p,
                     'mean': compute_mean,
                     'std': compute_std,
                     'skewness': compute_skewness,

                             'kurtosis': compute_kurtosis}
```
         

    - Frequency-domain features: the spectrum (magnitude-squared) of the Fourier transform modes
    - Label assignment is also done according to the mapping from the last section

### 2. FL dataset partition

The processed dataset is further partitioned into FL datasets. Each bearing corresponding to one FL client. Note: bearing number 1, 2, 3, 4 are renamed to client_ids 0, 1, 2, 3, for use of FL simulation. We can choose to specify the holdout bearings (defaults to bearing number 4) for the purpose of demonstrating the model inference step; these bearings will not participate in the FL training.


## The `nasa-bearings` container in `octaipipedatasets` blob storage

The processed NASA bearing dataset is stored in the `nasa-bearings` container in `octaipipedatasets` blob storage. The container has the following structure:

```
nasa-bearings
|
├── octaipipe-tutorials
    ├── client_0
    |   └── X_time.csv
    ├── client_1
    |   └── X_time.csv
    ├── client_2
    |   └── X_time.csv
    ├── client_3
    |   ├── X_time.csv
    |   └── y.csv
    ├── clusters_client_3.png
    └── record_scores.csv
├── bearings_1.csv
├── bearings_1.tar.gz
├── bearings_2.csv
├── bearings_2.tar.gz
├── bearings_3.csv
├── bearings_3.tar.gz
├── bearings_4.csv
├── bearings_4.tar.gz

```

The `bearings_<i>.csv` and `bearings_<i>.tar.gz` files are used in `Product-Demos/Demo-vX.X-Clustering`, whereas files in `octaipipe-tutorials` are used in the k-fed tutorial in OctaiPipe's documentation.



## Metadata

Each `bearings_<i>.csv` file contains the following columns:
```
_time, label, [<x_time/frequency_features>], [<y_time/frequency_features>]
```
where the classes in the `label` column are given above, and the x- and y-channel time/frequency domain feature columns total 422 columns.

The label column is popped out and the feature datasets are stored as `X_time.csv` in octaipipe-tutorials, for the k-fed clustering tutorial. `client_3` also has the ground truth target label stored in `y.csv`.

The number of rows in each bearing dataset are as follows:
- client_0 (=bearing_1): 2151
- client_1 (=bearing_2): 2152
- client_2 (=bearing_3): 2152
- client_3 (=bearing_4): 1836
