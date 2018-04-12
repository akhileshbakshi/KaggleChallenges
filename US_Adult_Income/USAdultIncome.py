# very crude attempt. potential improvements: 
# 1. classify categorical variables into fewer categories 
# 2. use xgboost! - currently not installed on system 


 # setting up workspace -------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier


# user-defined functions to aid data cleaning ---------------------------------

def func_addDumies (inData, targetColumns): 
# use categories in targetColumns to create new columns with binary data
    outData = pd.DataFrame() 
    for index, col in enumerate(targetColumns):       
        tempData = pd.get_dummies(inData[col] , prefix=col)
        outData[list(tempData)] = tempData
    return outData      



# import data -----------------------------------------------------------------
    
columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain','CapitalLoss',
           'HoursPerWeek','Country','Income']
train_set = pd.read_csv('adult-training.csv', names=columns)
test_set = pd.read_csv('adult-test.csv', names=columns, skiprows=1)

allData = train_set.append( test_set , ignore_index = True )
trainSamples = 32561
del train_set, test_set


# data engineering ------------------------------------------------------------

# drop irrelevant columns 
allData.drop(["fnlgwt", "Country", "Education"], axis=1, inplace=True)   

# replace all ? with unknown, and remove special characters 
for icol in list(allData): 
    allData[icol].replace(' ?', 'Unknown', inplace = True)
    if allData[icol].dtype != 'int64':
        allData[icol] = allData[icol].apply(lambda val: val.replace(" ", ""))
        allData[icol] = allData[icol].apply(lambda val: val.replace(".", ""))  

# binning ages in groups of 10 
labels = ["{0}-{1}".format(i, i + 9) for i in range(0,100,10)]
allData['AgeGroup'] = pd.cut(allData.Age, range(0, 101, 10), right=False, labels=labels)
allData.drop(["Age"], axis=1, inplace=True)
 
# binning EdNum in groups of 5 (this is indicative of education level)
labels = ["{0}-{1}".format(i, i + 4) for i in range(0, 20, 5)]
allData['Education'] = pd.cut(allData.EdNum, range(0, 21, 5), right=False, labels=labels)
allData.drop(["EdNum"], axis=1, inplace=True)

# mapping non-numeric data to numeric data and adding 
targetColumns =  ['AgeGroup','Workclass','Education','MaritalStatus',
           'Occupation','Relationship','Race','Sex','Income']
tempData = func_addDumies(allData,targetColumns)
allData[list(tempData)] = tempData
allData.drop(targetColumns, axis=1, inplace=True)

# removing unknowns/ redundant columns, should also group together similar professions  
targetColumns =  ['AgeGroup_0-9','AgeGroup_90-99','Workclass_Unknown','Occupation_Unknown',
           'Race_Other','Sex_Female','Income_>50K']
allData.drop(targetColumns, axis=1, inplace=True)



# creating train, test data ---------------------------------------------------
columns = list(allData)
columns.remove("Income_<=50K")

allX, allY = allData[columns].values, allData["Income_<=50K"].values

train_X = pd.DataFrame(data = allX[:trainSamples], columns = columns)
train_Y = pd.DataFrame(data = allY[:trainSamples], columns = ["Income_<=50K"]) 
test_X = pd.DataFrame(data = allX[trainSamples:], columns = columns)
test_Y = pd.DataFrame(data = allY[trainSamples:], columns = ["Income_<=50K"])



# random forest classification ------------------------------------------------ 
parameters = {
     'n_estimators':(100, 500),
     'max_depth':(None, 24),
     'min_samples_split': (4, 8),
     'min_samples_leaf': (4, 12)
}

ranForest = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=8, verbose = 10)
ranForest.fit(train_X, train_Y)
ranForest.best_score_, ranForest.best_params_
ranForest.score(test_X, test_Y)

ranForest = RandomForestClassifier(n_estimators=500, max_depth=24, min_samples_leaf=4, min_samples_split=4)
ranForest.fit(train_X, train_Y)
importances = ranForest.feature_importances_
indices = np.argsort(importances)[-12:]
outColumns = [columns[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), outColumns)
plt.xlabel('Relative Importance')



# adaboost classification -----------------------------------------------------
parameters = {
     'base_estimator__max_features':(6,10), # these belong to decision trees 
     'base_estimator__max_depth':(3, 6),
     'base_estimator__min_samples_split': (4, 8),
     'base_estimator__min_samples_leaf': (4, 12), 
     'n_estimators': (50, 200), 
     'learning_rate': (1, 0.1)

}

dtc = DecisionTreeClassifier()
adaBoost = GridSearchCV(AdaBoostClassifier(dtc), parameters, cv=4, n_jobs=4, verbose = 10)
adaBoost.fit(train_X, train_Y)
adaBoost.best_estimator_

dtc = DecisionTreeClassifier(max_features=6, max_depth=6, min_samples_leaf=12, min_samples_split=4)
adaBoost = AdaBoostClassifier(dtc, n_estimators=200, learning_rate=0.1)
adaBoost.fit(train_X, train_Y)
adaBoost.score(test_X, test_Y)




