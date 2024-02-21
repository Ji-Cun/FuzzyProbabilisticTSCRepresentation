# A Fuzzy-Probabilistic Representation Learning for Time Series Classification

This repository contains the code implementation for the paper:

**_A Fuzzy-Probabilistic Representation Learning Method for Time Series Classification_**  [doi: 10.1109/TFUZZ.2024.3364585](http://doi.org/10.1109/TFUZZ.2024.3364585)<br>
in **IEEE Transactions on Fuzzy Systems**





Please cite as:

  ```bibtex
@ARTICLE{erazocosta_etal_2024,
  author={Erazo-Costa, Fabricio Javier and Silva, Petrônio C. L. and Guimarães, Frederico Gadelha},
  journal={IEEE Transactions on Fuzzy Systems}, 
  title={A Fuzzy-Probabilistic Representation Learning Method for Time Series Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Time series analysis;Classification algorithms;Computational modeling;Rockets;Forecasting;Training;Data models;Time series classification;fuzzy time series;representation learning;fuzzy temporal patterns},
  doi={10.1109/TFUZZ.2024.3364585}}

  }

```

## Probabilistic Weighted Fuzzy Time Series as Features <br>

**Example**

This example creates the PWFTS representation for the time series Illustrative Example of the paper.

```python
TimeSeries = np.array([-2.060, 0.780, -0.550, 1.770, -0.310, -1.020])
PWFTSrepresentation = TS2PWFTS(TimeSeries=TimeSeries, npartitions=4)
print(PWFTSrepresentation)
```

## Classification Reproducibility

```python
numDataset = 23 # Choose from 1 to 24
option = 'RF' # Use 'RF' or 'SVM'
params = paramsSelection(numDataset, option)
name = params['name']
npartitions = params['npartitions']
typeWindow = params['typeWindow']

# Opening DataSet
# Adjust the file path in this example to match the location where the 'UCRArchive_2018' folder is stored 
path = 'C:/Users/nameUser/Documents/UCRArchive_2018/' + name + '/' # This is a path example and it needs to be changed
[training_data, test_data] = OpeningDataSet(path, name)

# Separate in traning, testing -> times series and labels
Labels_training, X_training = training_data[0].astype(np.int32), training_data.loc[:, training_data.columns != 0].values
Labels_test, X_test = test_data[0].astype(np.int32), test_data.loc[:, test_data.columns != 0].values
[NumObsTrain, NumSample] = X_training.shape
[NumObsTest, NumSample] = X_test.shape

# Compute PWFTS features
X_features_Train = PWFTSfeatures(X_training, NumObsTrain, NumSample, npartitions, typeWindow)
X_features_Test = PWFTSfeatures(X_test, NumObsTest, NumSample, npartitions, typeWindow)

# CLASSIFICATION
accuracyClassification = Classification(option, X_features_Train, Labels_training, X_features_Test, Labels_test, params)
print('Accuracy: ' + str(accuracyClassification))
```
