import numpy as np
import pandas as pd
import sqlalchemy
import logging
from damadicsDBMapping import ValveReading
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import and_
from copy import copy

class DataManagerDamadics:
	"""Used to manage the access to the data stored in the database"""

	def __init__(self, user="", password="", engineType="", dbName="", host="", port=""):
		"""Attempt to stablish a connection to the database"""

		#Object variables
		self.dbSession = None
		self.databaseString = engineType + user + ":" + password + "@" + host + ":" + port + "/" + dbName
		self.logger = logging.getLogger("damadics."+__name__)

		#setup logger
		self.logger.setLevel(logging.DEBUG)
		handler = logging.FileHandler("DataManager.log")
		handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		#Attempt connection to the database
		try:
			sqlengine = sqlalchemy.create_engine(self.databaseString)
			Session = sessionmaker(bind=sqlengine)
			self.dbSession = Session()

			self.logger.debug("Connection to " + engineType + host + ":" + port + "/" + dbName + " successfull")
		except Exception as e:
			self.logger.debug(traceback.format.exc())
			self.logger.debug("Error in connection to " + engineType + host + ":" + port + "/" + dbName)
			return False


	def getComponentReadingClassByType(self, componentType = "Valve"):
		"""Map each componentType to its corresponding reading class"""

		componentClass = None

		if componentType == "Valve":
			componentClass = ValveReading
		else:
			componentClass = None

		return componentClass


	def readData(self, startTimestamp, endTimestamp, componentTypes = []):
		"""Return a list of dataframes with the data for each of the components in componentTypes in the indicated period of time"""

		#THIS MUST BE PARALLELIZED IN ORDER TO ACHIEVE ITS MAXIMUM PERFORMANCE

		if self.dbSession == None:
			self.logger.error('Session not initialized')
			return

		dataFrames = dict()
		readings = dict()

		count = 0

		componentsReadingClass = list()

		for component in componentTypes:

			readingClass = self.getComponentReadingClassByType(component)
			if readingClass != None:
				componentsReadingClass.append(readingClass)

		#This goes to the log
		self.logger.debug('Loading data for components {} from {} to {}'.format(componentsReadingClass, startTimestamp, endTimestamp))

		#Create the dataframes for each component
		for componentReadingClass in componentsReadingClass:
			readings = \
			self.dbSession.query(componentReadingClass).filter(and_(componentReadingClass._timestamp >= startTimestamp, componentReadingClass._timestamp < endTimestamp)).all()

			if readings != []:
				#build the dictionary object
				dictionaryOfReadings = copy(readings[0]).__dict__
				#print(dictionaryOfReadings)
				dictionaryOfReadings.pop('_sa_instance_state')

				#This is performed fast, no need to change it
				for key in dictionaryOfReadings:
					dictionaryOfReadings[key] = list()

				keys = dictionaryOfReadings.keys()
				
				#Can we make this perform faster?
				for reading in readings:
					readingAsDict = reading.__dict__
					readingAsDict.pop('_sa_instance_state')

					#Can we write a map function (functional programming) for this?
					for key in keys:
						dictionaryOfReadings[key].append(readingAsDict[key])

				dataFrames[componentTypes[count] + 'Readings'] =  pd.DataFrame.from_dict(dictionaryOfReadings)

			count += 1

		return dataFrames


	def reshapeAndCleanDataFrame(self, dataFrame, removeSetpoints=False, removeRequests=False, removeBooleans=False):
		"""Reshape the dataframe and return the dataframe reshaped. For each dataframe keep only the data/features that may result useful for the analysis
		for instance it makes no sense to keep boolean values since they are not drawn from a gaussian distribution but instead a binary distribution, 
		and thus they can not be fitted into a gausssian one. Same with setpoints, they are fixed values."""

		#clean dataframe and reshape it
		df = dataFrame
		dropColumns = None

		newColumnNames = {column:column.replace('_', "") for column in df.columns}
		df.rename(columns=newColumnNames, inplace=True)
		
		#Get the Id Column
		idColumn = list(filter(lambda colName: 'id' in colName.lower() or 'number' in colName.lower(), newColumnNames.values()))
		if len(idColumn) < 1:
		    self.logger.error('Could not determine Id Column')
		else:
		    idColumn = idColumn[0]

		indexColumns = ['timestamp']

		dataColumns = list(filter(lambda colName: idColumn != colName and colName != 'timestamp', newColumnNames.values()))

		#Remove setpoint related columns
		if removeSetpoints == True:
			dropColumns = list(filter(lambda colName: 'setpoint' in colName.lower() or 'stpnt' in colName.lower(), newColumnNames.values()))

		#Remove request related columns
		if removeRequests == True:
			dropColumns += list(filter(lambda colName: "request" in colName.lower() or "req" in colName.lower(), newColumnNames.values()))

		#Remove boolean columns
		if removeBooleans == True:
			dropColumns += list(filter(lambda colName: df[colName].dtype == bool, newColumnNames.values()))

		#drop the undesired columns
		if dropColumns != None:
			df.drop(dropColumns, axis=1, inplace=True)
		
		df.set_index(indexColumns, inplace=True)
		df.dropna(axis=1, how='all', inplace=True) #drop null value columns

		currentColumns = df.columns.values.tolist()

		currentColumns.remove(idColumn)

		#when there is a -1 in the data, replace it by NaN
		#df.replace(-1, value=np.nan, inplace=True)
		df.dropna(axis=0, how='all', inplace=True, subset=currentColumns)
		#df.fillna(value=-1)

		#dataFrames[dfkey].fillna(value=nan, inplace=True)

		return df


	def retrieve_and_reshape_data(self, startDateTime, endDateTime, desiredComponents, columnArrangement):
	    """Gets the data and reshapes it to give me X and y as per machine learning standards.
	    For the labels, 0 means normal and 1 means faulty."""

	    dataFrames = self.readData(startDateTime, endDateTime, desiredComponents)
	    
	    features = ['disturbedMediumFlow', 'pressureValveInlet', 'pressureValveOutlet', 
	                'mediumTemperature', 'rodDisplacement', 'externalControllerOutput']
	    
	    df = dataFrames['ValveReadings']
	    df = self.reshapeAndCleanDataFrame(df)
	    df = df[columnArrangement] #Rearrange columns
	    
	    #display(df.head())
	    
	    X = df[features].values
	    #X = X.T
	    
	    selectedFault = df['selectedFault'].values
	    selectedFault = (selectedFault != 20) #20, in the database, is given to normal observations
	    y = selectedFault.astype(int)
	    y = np.reshape(y, (y.shape[0], 1))
	    #y = y.T
	    
	    return X, y, df


	def closeConnection(self):
		"""Close the session"""

		try:
			self.dbSession.close()
			self.logger.debug("Database connection successfully closed")
		except e:
			self.logger.error("Could not close database connection")






