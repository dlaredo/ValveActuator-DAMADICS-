import h5py
import numpy as np
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import pandas as pd
import datetime
import sys


sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')

from ann_framework.data_handlers.damadicsDBMapping import *


def connect_to_db(username, pasw, host, dbname):

	databaseString = "mysql+mysqldb://" + username + ":" + pasw + "@" + host + "/" + dbname


	try:
		sqlengine = sqlalchemy.create_engine(databaseString)
		SQLSession = sessionmaker(bind=sqlengine)
		sqlsession = SQLSession()
		print("Connection to " + databaseString + " successfull")
	except Exception as e:
		print("e:", e)
		print("Error in connection to the database")

	return sqlsession


def extract_data_from_db(sqlsession, start_date, end_date):
	y_col_name = 'selectedFault'

	print("Reading data from ValveReading")
	query = sqlsession.query(ValveReading).filter(ValveReading._timestamp.between(start_date, end_date))


	df_database = pd.read_sql(query.statement, sqlsession.bind)

	if df_database.shape[0] == 0:
		print("No data found for ValveReading dates between {} and {}. Aborting.\n".format(start_date, end_date))
		return -1


	return df_database



def main():

	"""
	['externalControllerOutput', 'undisturbedMediumFlow', 'pressureValveInlet', 'pressureValveOutlet',
	 'mediumTemperature', 'rodDisplacement', 'disturbedMediumFlow','selectedFault', 'faultType', 'faultIntensity']
	 """

	# Read .mat file
	f = h5py.File('/Users/davidlaredorazo/Documents/MATLAB/Damadics/SimulinkModel/DABLib/actuator_data.mat', 'r')
	data = f['actuator_data'][()]

	data = data[:, 1:]  # Remove data index

	data0 = data[0, :].reshape((1, data.shape[1]))
	data_rest = data[59::60]

	print(data)

	data_minutes_matlab = np.vstack((data0, data_rest))


	#Read database

	time_delta = datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=1, hours=0, weeks=0)
	start_date = datetime.datetime(2019, 7, 19, 19, 47, 43)  # Start date
	end_date = start_date + 20*time_delta


	sqlsession = connect_to_db('dlaredorazo', '@Dexsys13', '127.0.0.1', 'damadics2')
	df_db = extract_data_from_db(sqlsession, start_date, end_date)

	df_sub = df_db.loc[:,['externalControllerOutput', 'pressureValveInlet',
            'pressureValveOutlet', 'rodDisplacement', 'disturbedMediumFlow', 'mediumTemperature', 'faultIntensity']]

	#df_sub['selectedFault'] = df_sub['selectedFault'].map(lambda x: 0 if x == 20 else x)

	data_db = df_sub.values

	print(data_db.shape)
	print(data_db)

	data_minutes_matlab = data_minutes_matlab[:data_db.shape[0]]
	data_minutes_matlab = np.around(data_minutes_matlab, decimals=6)

	print(data_minutes_matlab.shape)
	print(data_minutes_matlab)

	residual = data_minutes_matlab - data_db

	print(np.sum(residual, axis=0))
	print(np.sum(residual))

main()