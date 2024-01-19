# Fabricio Javier Erazo-Costa , Petrônio C. L. Silva , and Frederico Gadelha Guimarães
# A Fuzzy-Probabilistic Representation Learning Method for Time Series Classification

import numpy as np
from numba import njit

@njit(fastmath = True)
def trimf(x, parameters):
    xx = np.round(x, 3)
    if xx < parameters[0]:
        return 0
    elif parameters[0] <= xx < parameters[1]:
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif parameters[1] <= xx <= parameters[2]:
        return (parameters[2] - xx) / (parameters[2] - parameters[1])
    else:
        return 0

@njit(fastmath = True)
def fuzzify(value,npartitions, partitions):
  f_n = np.zeros((npartitions))
  for k in range(npartitions):
    if value >= partitions[k][0] and value <= partitions[k][2]:
      f_n[k] =  trimf(value, partitions[k])
  return f_n

@njit(fastmath = True)
def TS2PWFTS(TimeSeries, npartitions):

  vmin = np.min(TimeSeries)
  vmax = np.max(TimeSeries)
  min=vmin-0.1*np.abs(vmin)
  max= vmax+0.1*np.abs(vmax)

  centers = np.linspace(min, max, npartitions)
  partlen = (max-min)/(npartitions - 1)

  partitions = np.zeros((npartitions,3))
  for i in range(npartitions):
    partitions[i][:] = [centers[i] - partlen, centers[i], centers[i] + partlen]

  ZAi = np.zeros((npartitions))
  FuzzyRulesMatrix = np.zeros((npartitions,npartitions))

  for Sample in range(len(TimeSeries)-1):
    f_n = fuzzify(TimeSeries[Sample],npartitions, partitions)
    f_n_1 = fuzzify(TimeSeries[Sample+1],npartitions, partitions)

    FuzzyRulesMatrix = FuzzyRulesMatrix + np.outer(f_n, f_n_1)

  ZAi = np.sum(FuzzyRulesMatrix, axis=1)

  ZAall = np.sum(FuzzyRulesMatrix)
  Pi = ZAi/ZAall

  FeaturesVector = np.zeros((npartitions * (npartitions + 1),))
  for Precedent in range(npartitions):

    if ZAi[Precedent] != 0:
      FeaturesVector[Precedent * (npartitions + 1)] = Pi[Precedent]
      FeaturesVector[Precedent * (npartitions + 1) + 1:(Precedent + 1) * (npartitions + 1)] = FuzzyRulesMatrix[Precedent] / ZAi[Precedent]
    elif ZAi[Precedent] == 0:
      FeaturesVector[Precedent * (npartitions + 1)] = Pi[Precedent]
      FeaturesVector[Precedent * (npartitions + 1) + 1:(Precedent + 1) * (npartitions + 1)] = FuzzyRulesMatrix[Precedent]

  return FeaturesVector

def PWFTSfeatures(TimeSeriesObservations, NumObs, NumSample, npartitions, typeWindow):
  # GLOBAL
  if typeWindow == 'Global' or typeWindow == 'Ensemble':
    X_features = np.zeros([NumObs,npartitions*(npartitions+1)])
    for i in range(NumObs):
      FeaturesVector = TS2PWFTS(TimeSeriesObservations[i], npartitions)
      X_features[i] =  FeaturesVector

    if typeWindow == 'Ensemble':
        X_features_Global = np.copy(X_features)
        del X_features

  # LOCAL
  if typeWindow == 'Local' or typeWindow == 'Ensemble':
    NumWindows = 4
    LengthWindows = NumSample / NumWindows
    X_features = np.zeros([NumObs,NumWindows*npartitions*(npartitions+1)])

    for i in range(NumObs):
      for nwindow in range(NumWindows):
        FeaturesVectorLocal = TS2PWFTS(TimeSeriesObservations[i, nwindow*int(LengthWindows):(nwindow+1)*int(LengthWindows)-1], npartitions)
        X_features[i, nwindow*npartitions*(npartitions+1):(nwindow+1)*npartitions*(npartitions+1)] = FeaturesVectorLocal

    if typeWindow == 'Ensemble':
      X_features_Local = np.copy(X_features)
      del X_features

  # ENSEMBLE
  if typeWindow == 'Ensemble':
    X_features = np.concatenate((X_features_Global, X_features_Local), axis=1)

  return X_features
