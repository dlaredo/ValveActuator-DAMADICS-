import dataManagement
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
from mpl_toolkits.mplot3d import Axes3D

global joinPlotVariables

joinPlotVariables = {"AHUReadings":['mixedAirTemperature', 'outsideAirTemperature', 'returnAirTemperature', 'supplyAirTemperature']}
selectedFeatures = {"AHUReadings":['mixedAirTemperature', 'outsideAirTemperature', 'returnAirTemperature', 'supplyAirTemperature']}

pca = PCA(n_components=3) #For visualization purposes only
scaler = StandardScaler()

np.set_printoptions(threshold=np.inf)

def getDesiredData(desiredComponents, startDateTime, endDateTime, removeSetpoints=False, removeRequests=False, removeBooleans=False):
	"""Get the data from the desired components at the specified dates. After retrieval, clean and reshape the dataframes
	to disregard unuseful data such as setpoints and requestpoints. Return the cleaned dataframes"""

	#Get the desired data and clean it
	dataFrames = dataManagement.loadData(startDateTime, endDateTime, desiredComponents)

	for key in dataFrames:
		df = dataFrames[key]
		dataFrames[key] = dataManagement.reshapeAndCleanDataFrame(df, removeSetpoints=removeSetpoints, removeRequests=removeRequests, removeBooleans=removeBooleans)

	return dataFrames


def plotData(dataFrame, titleString, saveToFile=False, nsamples = 100, vars=None):
	"""Generate plots of the data in dataframe. Right now only joinplots are generated"""

	#plot nsamples random elements from the sample
	random_elements = random.sample(range(dataFrame.shape[0]), nsamples)
	dataFrame_plot = dataFrame.iloc[random_elements]

	if vars != None:
		p = sns.pairplot(dataFrame_plot, hue='AHUNumber', diag_kind='kde', vars=vars, dropna=True)
	else:
		p = sns.pairplot(dataFrame_plot, hue='AHUNumber', diag_kind='kde', dropna=True)

	plt.suptitle(titleString, y = 1.025)

	if saveToFile == True:
		plt.savefig('testfig.png', format='png', pad_inches=0.5, bbox_inches='tight')
		plt.close()
	else:
		pass
		#plt.show()


def plotTrainedModel(classifier, X, nsamples = 1000):
	"""Plot the mahalanobis disntances of the trained model"""

	n_features = X.shape[1]

	"""
	linspaces = list()

	for i in range(n_features):
		linspaces.append(np.linspace(np.min(X[:, i])-5, np.max(X[:, i])+5, 5))

	grids = np.meshgrid(*linspaces)
	gridsRavel = [grid.ravel() for grid in grids]
	Xgrid = np.column_stack(gridsRavel)  # Feature matrix containing all grid points
	decision = classifier.decision_function(Xgrid)
	print(decision)
	"""

	random_elements = random.sample(range(X.shape[0]), nsamples)
	X_randomSamples = X[random_elements]

	inliers = list()
	outliers = list()

	labels = classifier.predict(X)
	label_randomSamples = labels[random_elements]

	for i in range(len(labels)):
		if labels[i] == 1:
			inliers.append(X[i])
		else:
			outliers.append(X[i])

	inliers = np.array(inliers)
	outliers = np.array(outliers)

	print("Number of total inliers " + str(len(inliers)))
	print("Number of total outliers " + str(len(outliers)))

	inliers = list()
	outliers = list()

	for i in range(len(label_randomSamples)):
		if label_randomSamples[i] == 1:
			inliers.append(X_randomSamples[i])
		else:
			outliers.append(X_randomSamples[i])

	inliers = np.array(inliers)
	outliers = np.array(outliers)

	print("Number of sampled inliers " + str(len(inliers)))
	print("Number of sampled outliers " + str(len(outliers)))

	X2 = scaler.fit_transform(X_randomSamples)
	X_reduce = pca.fit_transform(X2)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# Plot the compressed data points
	ax.scatter(inliers[:, 0], inliers[:, 1], zs=inliers[:, 2], s=4, lw=0, label="inliers")

	# Plot x's for the ground outliers
	ax.scatter(outliers[:, 0], outliers[:, 1], zs=outliers[:, 2], lw=2, s=60, marker="x", c="red", label="outliers")
	ax.set_xlabel('Coordinate x')
	ax.set_ylabel('Coordinate y')
	ax.set_zlabel('Coordinate z')
	ax.legend()

	plt.show()


def main():

	classifiers = {'AHUReadings':EllipticEnvelope(contamination=0.035)}

	pd.options.mode.chained_assignment = None

	nsamples = 1000

	desiredComponents = ['AHU']

	startDateTime = datetime(2017, 1, 1, hour=0, minute=0, second=0, microsecond=0)
	endDateTime = datetime(2017, 3, 1, hour=0, minute=0, second=0, microsecond=0)

	dataFrames = getDesiredData(desiredComponents, startDateTime, endDateTime, removeSetpoints=True, removeRequests=True, removeBooleans=True)

	titleString = "Comparisson of the four measured temperatures (" + str(nsamples) + " random samples)\n by the AHU from " + str(startDateTime) + " to " + str(endDateTime)

	for key in dataFrames:

		#Perform the feature selection, selected features are chosen manually and clean the dataset for performing plotting tasks
		df = dataFrames[key]
		idColumn = list(filter(lambda colName: 'Id' in colName or 'Number' in colName, df.columns.values))
		dfSelectedFeatures = df[selectedFeatures[key] + idColumn]
		dfSelectedFeatures.dropna(inplace=True)
		#print(dfSelectedFeatures.columns)

		#Generate a pairplot of the data used for training the model
		plotData(dfSelectedFeatures, titleString, saveToFile=True, nsamples=nsamples, vars=joinPlotVariables[key])
		
		#Train the classifier with the selected features only
		classifier = classifiers[key]
		dfSelectedFeatures = dfSelectedFeatures[selectedFeatures[key]]
		X = dfSelectedFeatures.values
		classifier.fit(X)

		#For the train set perform prediction
		#prediction = classifier.predict(X)
		print(classifier.threshold_)

		plotTrainedModel(classifier, X, nsamples=nsamples)

main()






