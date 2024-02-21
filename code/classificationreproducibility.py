def paramsSelection(num, option):
    if num == 1:
        if option == 'RF': params = {'name':'BeetleFly','npartitions': 7, 'typeWindow': 'Global', 'typeBalanced': None}
        elif option == 'SVM': params = {'name': 'BeetleFly', 'npartitions': 10, 'typeWindow': 'Local','typeBalanced': None,'gamma': 0.001, 'C': 10000}

    elif num == 2:
        if option == 'RF': params = {'name': 'Computers','npartitions': 20, 'typeWindow': 'Global', 'typeBalanced': None}
        elif option == 'SVM': params = {'name': 'Computers', 'npartitions': 30, 'typeWindow': 'Ensemble','typeBalanced': None, 'gamma': 0.01, 'C': 10}

    elif num == 3:
        if option == 'RF': params = {'name': 'DiatomSizeReduction', 'npartitions': 5, 'typeWindow': 'Global', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'DiatomSizeReduction', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.01, 'C': 10000}

    elif num == 4:
        if option == 'RF': params = {'name': 'DistalPhalanxOutlineAgeGroup', 'npartitions': 10, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'DistalPhalanxOutlineAgeGroup', 'npartitions': 10, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 1}

    elif num == 5:
        if option == 'RF': params = {'name': 'DistalPhalanxOutlineCorrect', 'npartitions': 10, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'DistalPhalanxOutlineCorrect', 'npartitions': 6, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 10}

    elif num == 6:
        if option == 'RF': params = {'name': 'DistalPhalanxTW', 'npartitions': 8, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'DistalPhalanxTW', 'npartitions': 30, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 0.01, 'C': 10}

    elif num == 7:
        if option == 'RF': params = {'name': 'Earthquakes', 'npartitions': 8, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Earthquakes', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 960.9220764, 'C': 1676067626}

    elif num == 8:
        if option == 'RF': params = {'name': 'ECG5000', 'npartitions': 5, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'ECG5000', 'npartitions': 8, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 10}

    elif num == 9:
        if option == 'RF': params = {'name': 'GunPoint', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'GunPoint', 'npartitions': 15, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 1}

    elif num == 10:
        if option == 'RF': params = {'name': 'Herring', 'npartitions': 7, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Herring', 'npartitions': 15, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.01, 'C': 100}

    elif num == 11:
        if option == 'RF': params = {'name': 'Meat', 'npartitions': 20, 'typeWindow': 'Local', 'typeBalanced': None}
        elif option == 'SVM': params = {'name': 'Meat', 'npartitions': 20, 'typeWindow': 'Ensemble', 'typeBalanced': None, 'gamma': 1, 'C': 1}

    elif num == 12:
        if option == 'RF': params = {'name': 'MiddlePhalanxOutlineAgeGroup', 'npartitions': 6, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'MiddlePhalanxOutlineAgeGroup', 'npartitions': 30, 'typeWindow': 'Global', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 1}

    elif num == 13:
        if option == 'RF': params = {'name': 'MiddlePhalanxOutlineCorrect', 'npartitions': 8, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'MiddlePhalanxOutlineCorrect', 'npartitions': 6, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 1, 'C': 100}

    elif num == 14:
        if option == 'RF': params = {'name': 'MiddlePhalanxTW', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'MiddlePhalanxTW', 'npartitions': 10, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 1, 'C': 10}

    elif num == 15:
        if option == 'RF': params = {'name': 'Plane', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Plane', 'npartitions': 30, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.01, 'C': 10}

    elif num == 16:
        if option == 'RF': params = {'name': 'ProximalPhalanxOutlineAgeGroup', 'npartitions': 7, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'ProximalPhalanxOutlineAgeGroup', 'npartitions': 5, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 1, 'C': 10}

    elif num == 17:
        if option == 'RF': params = {'name': 'ProximalPhalanxOutlineCorrect', 'npartitions': 7, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'ProximalPhalanxOutlineCorrect', 'npartitions': 6, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 1, 'C': 10}

    elif num == 18:
        if option == 'RF': params = {'name': 'ProximalPhalanxTW', 'npartitions': 5, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'ProximalPhalanxTW', 'npartitions': 6, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.01, 'C': 1000}

    elif num == 19:
        if option == 'RF': params = {'name': 'ShapeletSim', 'npartitions': 8, 'typeWindow': 'Global', 'typeBalanced': None}
        elif option == 'SVM': params = {'name': 'ShapeletSim', 'npartitions': 7, 'typeWindow': 'Global', 'typeBalanced': None, 'gamma': 10, 'C': 10}


    elif num == 20:
        if option == 'RF': params = {'name': 'SonyAIBORobotSurface1', 'npartitions': 7, 'typeWindow': 'Global', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'SonyAIBORobotSurface1', 'npartitions': 5, 'typeWindow': 'Global', 'typeBalanced': 'balanced', 'gamma': 10, 'C': 10}

    elif num == 21:
        if option == 'RF': params = {'name': 'Strawberry', 'npartitions': 20, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Strawberry', 'npartitions': 10, 'typeWindow': 'Local', 'typeBalanced': 'balanced', 'gamma': 0.1, 'C': 100}

    elif num == 22:
        if option == 'RF': params = {'name': 'Trace', 'npartitions': 5, 'typeWindow': 'Global', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Trace', 'npartitions': 5, 'typeWindow': 'Global', 'typeBalanced': 'balanced', 'gamma': 10, 'C': 0.1}

    elif num == 23:
        if option == 'RF': params = {'name': 'TwoLeadECG', 'npartitions': 5, 'typeWindow': 'Local', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'TwoLeadECG', 'npartitions': 30, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 0.0001, 'C': 1000}

    elif num == 24:
        if option == 'RF': params = {'name': 'Wafer', 'npartitions': 40, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced_subsample'}
        elif option == 'SVM': params = {'name': 'Wafer', 'npartitions': 40, 'typeWindow': 'Ensemble', 'typeBalanced': 'balanced', 'gamma': 0.001, 'C': 100}

    return params

import pandas as pd
def OpeningDataSet(path, name):

    file = path + name + '_TRAIN.tsv'
    training_data = pd.read_csv(file, header=None, delim_whitespace=True)
    file2 = path + name + '_TEST.tsv'
    test_data = pd.read_csv(file2, header=None, delim_whitespace=True)

    return training_data, test_data

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
def Classification(option, X_features_Train, Labels_training, X_features_Test, Labels_test, params):
    if option == 'RF':
        typeBalanced = params['typeBalanced']
        model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0, class_weight=typeBalanced)

    elif option == 'SVM':
        gamma = params['gamma']
        C = params['C']
        typeBalanced = params['typeBalanced']
        model = svm.SVC(kernel="rbf", gamma=gamma, C=C, class_weight=typeBalanced)


    model.fit(X_features_Train, Labels_training)
    predictedLabels = model.predict(X_features_Test)
    accuracyClassification = accuracy_score(Labels_test, predictedLabels) * 100

    return accuracyClassification
