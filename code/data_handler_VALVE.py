import numpy as np
import random
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select
from sqlalchemy import and_
from sqlalchemy import between
from sqlalchemy.sql import exists

from sqlalchemy import desc

from datetime import datetime, timezone, timedelta

from damadicsDBMapping import *
from sequenced_data_handler import SequenceDataHandler

class ValveDataHandler(SequenceDataHandler):

	## TODO: column information here

	"""
	ValveDataHandler applies data preprocessing steps to the raw data of the simulated
	valve dataset and format it in order to feed it to a machine learning model of your choice.

	Argument(s)
		start_time(string): MySQL DATETIME type of format 'YYYY-MM-DD HH:MM:SS'. Example: '1000-01-01 00:00:00'
		end_time(string): MySQL DATETIME type of format 'YYYY-MM-DD HH:MM:SS'. Example: '9999-12-31 23:59:59'
		selected_features(int): array of integers that indicate column(s) to extract
		sequence_length(int): (see superclass SequenceDataHandler)
		sequence_stride(int): (see superclass SequenceDataHandler)
		data_scaler: TODO (?)

	Attribute(s)
		self._start_time(string): starting time of the data collection from the database
		self._end_time(string): ending time of the data collection from the database
		self._selected_features(int): columns selected from the database to be used as inputs for the model
		self._rectify_labels(boolean): TODO (?)
		self._sqlsession: MySQL session (?)

		self._data_scaler: data normalization applied to the data set
		self._load_from_db(boolean): indicator of whether data has already been loaded into memory
		self._column_names(string): labels for each column that exists in the database
		self._df(pandas DataFrame): holds all the extracted data from the database (both inputs and output)
		self._X(list): holds all the input data
		self._y(list): holds all the output data
	"""

	#Method definition
	def __init__(self, start_time, end_time, selected_features, sequence_length = 1, sequence_stride = 1, data_scaler = None, problem = None):

		#Public Properties
		self._start_time = start_time
		self._end_time = end_time
		self._selected_features = selected_features
		self._rectify_labels = False
		self._sqlsession = None
		self._num_samples = None
		self._num_good = list()
		self._num_bad = list()

        
		if (problem == 'classification' or problem == 'regression' or problem == 'anomaly'):
			self._problem = problem
		else:
			print("Error on 'problem' parameter. Must be either 'anomaly', 'classification' or 'regression'.")
			return


		#ReadOnly Properties
		self._data_scaler = data_scaler
		self._load_from_db = True

		self._column_names = {0: 'timestamp', 1: 'externalControllerOutput', 2: 'undisturbedMediumFlow', 3: 'pressureValveInlet', 4:'pressureValveOutlet',
							  5: 'mediumTemperature', 6: 'rodDisplacement', 7: 'disturbedMediumFlow', 8: 'selectedFault', 9: 'faultType', 10: 'faultIntensity'}

		# Entire Dataset saved in memory
		self._df = None
		self._X = None
		self._y = None

		print("init")

		# Super Init
		super().__init__(sequence_length, sequence_stride, len(selected_features), data_scaler)

	# Public
	def connect_to_db(self, username, pasw, host, dbname):

		"""
		Creates a connection with a running database.

		Argument(s):
			username(string): username required to access the database
			pasw(string): password for the specified username
			host(string): (?)
			dbname(string): name of the database you would like to connect
		"""

		self.dbname = dbname
		databaseString = "mysql+mysqldb://"+username+":"+pasw+"@"+host+"/"+dbname

		self._sqlsession = None
		try:
			sqlengine = sqlalchemy.create_engine(databaseString)
			SQLSession = sessionmaker(bind=sqlengine)
			self._sqlsession = SQLSession()
			print("Connection to " + databaseString + " successfull")
		except Exception as e:
			print("e:", e)
			print("Error in connection to the database")


	# Private
	def extract_data_from_db(self):

		""" Extracts data on the valve dataset based on the start time and end time provided by the user. """

		startTime = datetime.now()

		# MySQL Query
		self._df = self._sqlsession.query(ValveReading).filter(ValveReading.timestamp.between (self._start_time,self._end_time) )
		self._df = pd.read_sql(self._df.statement, self._df.session.bind)

		# TODO: Grab self._selected_features
		# Assumption that the output is only one column and is located at the last column out of all the selected features
		self._X = self._df.loc[self._selected_features].values
		if (self._problem == 'classification'):
			vHandler._df['status'] = vHandler._df['selectedFault'].apply(lambda valve: 0 if valve == 20 else 1)
			self._y = self._df["status"].values
		elif (self._problem == 'regression'):
			self._y = self._df["selectedFault"].values

		#self._y = self._df["selectedFault"].values

		print("Extracting data from database runtime:", datetime.now() - startTime)


	# Private
	def one_hot_encode(self, num_readings):

		"""
		Custom one-hot-encoder for the type of valve fault.

		Argument(s):
			num_readings(int): row size of extracted dataset

		Return(s):
			one_hot_matrix(numpy matrix): one-hot-matrix representation of the output column with shape [num_readings, 20]
		"""

		startTime = datetime.now()

		fault_column = list()
		fault_column = self._y

		one_hot_matrix = np.zeros((num_readings, 20))
		for i in range(num_readings):
			one_hot_matrix[i, int(fault_column[i] - 1)] = 1

		return one_hot_matrix

	# Private
	def find_samples(self, data_samples):

		"""
		Splits up the extracted data into the different valve samples.
		Each sample should have the pattern of having consecutive 20s, which signifies the valve
		as not broken, and non-20s, meaning a fault has been detected. The non-20 fault MUST BE THE SAME for each sample.

		 	Example: [20, 20, 20, 20, 20, 20, 20, 8, 8, 8, 8, 8, 8]
					 [20, 20, 20, ... , 14, 14, 14, 14, ...., 14]

		Arguments
			data_samples(pandas DataFrame): data input or output (X or y)

		Return(s)
			big_list(list) *for inputs*: list of numpy arrays of shape (number of samples, amount of readings in each valve sample, number of selected features)
			big_list(list) *for outputs*: list of floats of shape (number of samples, amount fo readings in each valve sample, 1)
			counter(int): number of samples found in the dataset
		"""

		# TODO: handle cases when the first readings start of as a broken value
		# TODO: add a minimum amount of valves required

		startTime = datetime.now()

		small_list, big_list = list(), list()
		# Flag
		isBroken = False
		num_valves = 0

        
		if (self._problem == 'anomaly'):
			normal_status = 1.0
		elif (self._problem == 'classification'):
			normal_status = 0.0
		else:
			normal_status = 20.0
            
		num_good_readings = 0
		num_bad_readings = 0

		self._num_good = list()
		self._num_bad = list()


		for i in range(len(self._y)):
			# Case: Previous reading states Valve is broken
			if (isBroken):
				# Find that the current reading indicates that the Valve is not broken, meaning this is a new valve 
				if (self._y[i] == normal_status):
					num_good_readings += 1
					self._num_bad.append(num_bad_readings)
					num_bad_readings = 0

					isBroken = False
					num_valves += 1

					# Save completed valve reading
					small_list = np.vstack(small_list)
					big_list.append(small_list)
					small_list = list()
				else:
					num_bad_readings += 1
					small_list.append(data_samples[i])
				
			# Case: Previous reading states Valve is broken
			else:
				# Find that the current reading indicates that the Valve is now broken        
				if (self._y[i] != normal_status):
					num_bad_readings += 1
					self._num_good.append(num_good_readings)
					num_good_readings = 0

					isBroken = True
				else:
					num_good_readings += 1
					small_list.append(data_samples[i])


		print("Splitting into samples:",datetime.now() - startTime)
		print("counter:", num_valves)

		return big_list, num_valves


	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, test_ratio = 0, unroll = True):
		"""
		Load the data either from the database or from memory. Extracted data is then split into its training,
		cross-validation, and test sets as well as based on the specified parameters. The training, cross validation,
		and test lists are then reformatted so that the machine learning model training/testing can begin.

		Argument(s)
			verbose(int): custom print statement for data extraction
			cross_validation_ratio(float): ratio of the dataset that is put into the cross validation set
			test_ratio(float): ratio of the dataset that is put into the test set
			unroll(boolean): (see superclass SequenceDataHandler)

		"""

		if verbose == 1:
			print("Loading data for dataset {} with window_size of {}, stride of {}. Cros-Validation ratio {}".format(self._dataset_number,
				self._sequence_length, self._sequence_stride, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if test_ratio < 0 or test_ratio > 1:
			print("Error, test ratio must be between 0 and 1")
			return

		if cross_validation_ratio + test_ratio > 1:
			print("Sum of cross validation and test ratios is greater than 1. Need to pick smaller ratios.")
			return

		if self._load_from_db == True:
			print("Loading data from database")

			# Data extraction begins
			# self.extract_data_from_db()
            
			# Finds valve samples in the dataset
			self._X, self._num_samples = self.find_samples(self._X)

			# One hot encode output (if necessary)
# 			if (self._problem == 'regression' ):
# 				output_one_hot_matrix = self.one_hot_encode(self._df.shape[0])
# 				self._y, _ = self.find_samples(output_one_hot_matrix)
# 			else:
# 				self._y, _ = self.find_samples(self._y)
			self._y, _ = self.find_samples(self._y)

		else:
			print("Loading data from memory")

		#Reset arrays
		"""
		self._X_train_list = list()
		self._X_crossVal_list = list()
		self._X_test_list = list()
		self._y_train_list = list()
		self._y_crossVal_list = list()
		self._y_test_list = list()
		"""

		# Split up the data into its different samples
		# Modify properties in the parent class, and let the parent class finish the data processing
		self.train_cv_test_split(cross_validation_ratio, test_ratio)
		self.print_sequence_shapes()

		# Reformat split up data in order to create the training and tests sets to feed to the model
		# Unroll = True for ANN; Unroll = False for RNN
		self.generate_train_data(unroll)
		self.generate_crossValidation_data(unroll)
		self.generate_test_data(unroll)
        #
		self._load_from_db = False # As long as the dataframe doesnt change, there is no need to reload from file


	# Private
	def train_cv_test_split(self, cross_validation_ratio, test_ratio):

		"""
		Takes in the lists generated from splitting the data extracted into their different samples
		and splits them into the training, cross-validation, and test sets

		Argument(s)
			cross_validation_ratio(float): ratio of the dataset that is put into the cross validation set
			test_ratio(float): ratio of the dataset that is put into the test set
			num_samples(int): number of samples in the extracted dataset
		"""

		startTime = datetime.now()

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if test_ratio < 0 or test_ratio > 1:
			print("Error, test ratio must be between 0 and 1")
			return

		if cross_validation_ratio != 0 or test_ratio != 0:
			self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list, self._X_test_list, self._y_test_list = self.split_samples(cross_validation_ratio, test_ratio)

		print("Train, cv, and test splitting:",datetime.now() - startTime)


	# Private
	def split_samples(self, cross_validation_ratio, test_ratio):
		"""
		Split the samples according to their respective ratios

		Argument(s)
			cross_validation_ratio(float): ratio of the dataset that is put into the cross validation set
			test_ratio(float): ratio of the dataset that is put into the test set
			num_samples(int): number of samples in the extracted dataset

		Return(s)
			X_train_list(list): list of numpy arrays that holds reshuffled inputs of the training set
			y_train_list(list): list of numpy arrays that holds reshuffled outputs of the training set
			X_crossVal_list(list): list of numpy arrays that holds reshuffled inputs of the cross validation set
			y_crossVal_list(list): list of numpy arrays that holds reshuffled outputs of the cross validation set
			X_test_list(list): list of numpy arrays that holds reshuffled inputs of the test set
			y_test_list(list): list of numpy arrays that holds reshuffled outputs of the test set
		"""

		shuffled_samples = list(range(0, self._num_samples))
		random.shuffle(shuffled_samples)

		num_crossVal = int(cross_validation_ratio * self._num_samples)
		num_test = int(test_ratio * self._num_samples)
		num_train = self._num_samples - num_crossVal - num_test

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()
        
		train_defect_counter = 0
		crossVal_defect_counter = 0
		test_defect_counter = 0

		for i in range(num_train):
			X_train_list.append(self._X[shuffled_samples[i]])
			y_train_list.append(self._y[shuffled_samples[i]])
        
		# TODO: combine for loop for cross-validation set and test set
		for j in range(num_train, num_train + num_crossVal):
            
			random_index = np.random.randint(len(self._X[shuffled_samples[j]]) - self._sequence_length)
			X_crossVal_list.append(self._X[shuffled_samples[j]][random_index:random_index + self._sequence_length])
            
			if (self._problem == 'classification' or self._problem == 'anomaly'):
				y_crossVal_list.append(self._y[shuffled_samples[j]][random_index + self._sequence_length])
				if (y_crossVal_list[-1] == 1.0):
					crossVal_defect_counter += 1
			else:
				y_crossVal_list.append(self._y[shuffled_samples[j]][random_index + self._sequence_length].reshape(1, 20))
				if (y_crossVal_list[-1][-1][-1] == 1.0):
					crossVal_defect_counter += 1
                
		for k in range(num_train + num_crossVal, self._num_samples):
                                       
			random_index = np.random.randint(len(self._X[shuffled_samples[k]]) - self._sequence_length)                    
			X_test_list.append(self._X[shuffled_samples[k]][random_index:random_index + self._sequence_length])
                                       
			if (self._problem == 'classification' or self._problem == 'anomaly'):
				y_test_list.append(self._y[shuffled_samples[k]][random_index + self._sequence_length])
				if (y_test_list[-1] == 1.0):
					test_defect_counter += 1
			else:
				y_test_list.append(self._y[shuffled_samples[k]][random_index + self._sequence_length].reshape(1, 20))
				if (y_test_list[-1][-1][-1] == 1.0):
					test_defect_counter += 1
                
		print('Number of defective valves in cross-validation set: {} out of {}.'.format(crossVal_defect_counter, len(y_crossVal_list)))
		print('Number of defective valves in test set: {} out of {}.\n'.format(test_defect_counter, len(y_test_list)))        

		return X_train_list, y_train_list, X_crossVal_list, y_crossVal_list, X_test_list, y_test_list

	# ReadOnly Properties

	@property
	def df(self):
		return self._df

	@property
	def X(self):
		return self.X

	@property
	def y(self):
		return self._y

	@property
	def start_time(self):
		return self._start_time

	@property
	def end_time(self):
		return self._end_time

	@property
	def selected_features(self):
		return self._selected_features

	@property
	def sqlsession(self):
		return self._sqlsession

	@property
	def rectify_labels(self):
		return self._rectify_labels

	# Getters
	@start_time.setter
	def start_time(self,start_time):
		self._start_time = start_time

	@start_time.setter
	def end_time(self, end_time):
		self._end_time = end_time

	@selected_features.setter
	def selected_features(self, selected_features):
		self._selected_features = selected_features

	@rectify_labels.setter
	def rectify_labels(self, rectify_labels):
		self._rectify_labels = rectify_labels

	@sqlsession.setter
	def sqlsession(self,sqlsession):
		self._sqlsession = sqlsession

	# Custom string print
	def __str__(self):
		return "<ValveReading(timestamp='%s',externalControllerOutput='%s',undisturbedMediumFlow='%s',pressureValveInlet='%s',pressureValveOutlet='%s',mediumTemperature='%s',\
		rodDisplacement='%s',disturbedMediumFlow='%s',selectedFault='%s',faultType='%s',faultIntensity='%s')>"\
		%(str(self._timestamp),self._externalControllerOutput,self._undisturbedMediumFlow,self.pressureValveInlet,\
		self.pressureValveOutlet,self.mediumTemperature,self.rodDisplacement,self.disturbedMediumFlow,self.selectedFault,\
		self.faultType,self.faultIntensity)
