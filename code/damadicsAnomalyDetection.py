import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from dataManagement import DataManagerDamadics

from IPython.display import display, HTML
import plottingTools as pt


def main():


	#Variables setting
    columnArrangement = ['id', 'selectedFault', 'faultType', 'faultIntensity', 'externalControllerOutput', 
    'pressureValveInlet', 'pressureValveOutlet', 'disturbedMediumFlow', 'mediumTemperature', 'rodDisplacement']
    plottingVariables = ['externalControllerOutput', 'pressureValveInlet', 'pressureValveOutlet', 'disturbedMediumFlow', 
    'mediumTemperature', 'rodDisplacement']
    
    ellipticEnvelopeContamination = 0.04
    classifiers = {'DummyClf':DummyClassifier(), 'EllipticEnvelope':EllipticEnvelope(contamination=ellipticEnvelopeContamination)}

    pd.options.mode.chained_assignment = None

    nsamples = 1000
   
    random_seed = 0 #Change this to make it really random, 0 for testing purposes

    cv_folds = 4

    desiredComponents = ['Valve']

    scoringMetrics = ['precision_macro', 'recall_macro', 'f1_macro']

    startDateTime = datetime(2017, 11, 6, hour=0, minute=0, second=0, microsecond=0)
    endDateTime = datetime(2017, 11, 16, hour=0, minute=0, second=0, microsecond=0)

    dataManager = DataManager(user="readOnly", password="readOnly", engineType="mysql+mysqldb://", dbName="damadics", host="localhost", port="3306")

    y_trains = {'DummyClf':None, 'EllipticEnvelope':list()}
    y_tests = {'DummyClf':None, 'EllipticEnvelope':list()}

    #Data acquisition and formatting
    dataFrames = dataManager.readData(startDateTime, endDateTime, desiredComponents)
    
    df = dataFrames['ValveReadings']
    df = dataManager.reshapeAndCleanDataFrame(df)
    df = df[columnArrangement] #Rearrange columns
    
    #display(df.head())
    
    X_raw = df[['externalControllerOutput', 'disturbedMediumFlow', 'pressureValveInlet', 'pressureValveOutlet', 
        	'mediumTemperature', 'rodDisplacement']]
    df.loc[df['selectedFault'] != 20, 'selectedFault'] = 1
    df.loc[df['selectedFault'] == 20, 'selectedFault'] = 0

    totalCount = df.shape[0]
    faultydf = df.loc[df['selectedFault'] == 1, 'selectedFault']
    faultCount = faultydf.shape[0]
    nonFaultCount = df.shape[0] - faultCount
    faultNonFaultRatio = faultCount/nonFaultCount

    y_raw = df['selectedFault']
    #get a jointplot of the 7 variables
    """pt.jp_plotData(df, 'The 7 variables in the data', saveToFile='snspp_damadics.png', nsamples = 1000, 
    	vars=plottingVariables, hue='selectedFault')"""

    #Anomaly detection 

    #First standardize the data
    X_transformed = StandardScaler().fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_raw, random_state=random_seed)

    y_trains['EllipticEnvelope'] = [-1 if y == 1 else 1 for y in y_train]
    y_tests['EllipticEnvelope'] = [-1 if y == 1 else 1 for y in y_test]

    print('Performing cross validations')

    for classifierKey in classifiers:

    	print('\nResults for {} classifier'.format(classifierKey))

    	clf = classifiers[classifierKey]

    	if y_trains[classifierKey] != None:
    		y_train = y_trains[classifierKey]

    	cv_scores = cross_validate(clf, X_train, y_train, scoring=scoringMetrics, cv=cv_folds)
    	"""clf.fit(X_train, y_train)
    	y_pred = clf.predict(X_train)
    	score_acc = accuracy_score(y_train, y_pred)
    	print('accuracy {}'.format(score_acc))
    	print('Type: {}, first 5 elements {}, element type {}'.format(type(y_pred), y_pred[:5], type(y_pred[0])))"""

    	print('{}-fold cross validation'.format(cv_folds))

    	for key in cv_scores:
    		print("For metric %s Accuracy: %0.5f (+/- %0.5f)" % (key, cv_scores[key].mean(), cv_scores[key].std() * 2))
    
    print('\nTotal sample size {}, Train Size {}, Test Size {}'.format(X_raw.shape[0], X_train.shape[0], X_test.shape[0]))
    print('Total sample size {}, Faulty samples {}, Normal samples {}, Fault/Non Fault Ratio {:.4f}'.
    	format(totalCount, faultCount, nonFaultCount, faultNonFaultRatio))
    #print('Total sample size {}, Train Size {}, Test Size {}'.format(X_raw.shape[0], X_train.shape[0], X_test.shape[0]))
    


    #display((df))
    
    
    
    dataManager.endDataManager()


main()