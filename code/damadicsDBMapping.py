from sqlalchemy import Column, Integer, String, Table, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ValveReading(Base):
	"""Class to map to the DataPoints table in the HVAC DB"""

	__tablename__ = 'valvereadings'

	_id = Column('id', Integer, primary_key = True)
	_timestamp = Column('timestamp', DateTime, unique=True)
	_externalControllerOutput = Column('externalControllerOutput', Float, nullable=True)
	_undisturbedMediumFlow = Column('undisturbedMediumFlow', Float, nullable=True)
	_pressureValveInlet = Column('pressureValveInlet', Float, nullable=True)
	_pressureValveOutlet = Column('pressureValveOutlet', Float, nullable=True)
	_mediumTemperature = Column('mediumTemperature', Float, nullable=True)
	_rodDisplacement = Column('rodDisplacement', Float, nullable=True)
	_disturbedMediumFlow = Column('disturbedMediumFlow', Float, nullable=True)
	_selectedFault = Column('selectedFault', Integer, nullable=True)
	_faultType = Column('faultType', Integer, nullable=True)
	_faultIntensity = Column('faultIntensity', Integer, nullable=True)

	#Constructor

	def __init__(self, elementId, timestamp, externalControllerOutput = None, undisturbedMediumFlow = None, pressureValveInlet = None, pressureValveOutlet = None, 
		mediumTemperature = None, rodDisplacement = None, disturbedMediumFlow = None, selectedFault = None, faultType = None, faultIntensity = None):

		self._id = elementId
		self._timestamp = timestamp
		self._externalControllerOutput = externalControllerOutput
		self._undisturbedMediumFlow = undisturbedMediumFlow
		self._pressureValveInlet = pressureValveInlet
		self._pressureValveOutlet = pressureValveOutlet
		self._mediumTemperature = mediumTemperature
		self._rodDisplacement = rodDisplacement
		self._disturbedMediumFlow = disturbedMediumFlow
		self._selectedFault = selectedFault
		self._faultType = faultType
		self._faultIntensity = faultIntensity

	#Properties

	@property
	def elementId(self):
		return self._id

	@elementId.setter
	def elementId(self, value):
		self._id = value

	@property
	def timestamp(self):
		return self._timestamp

	@timestamp.setter
	def timestamp(self, value):
		self._timestamp = value

	@property
	def externalControllerOutput(self):
		return self._externalControllerOutput

	@externalControllerOutput.setter
	def externalControllerOutput(self, value):
		self._externalControllerOutput = value

	@property
	def undisturbedMediumFlow(self):
		return self._undisturbedMediumFlow

	@undisturbedMediumFlow.setter
	def undisturbedMediumFlow(self, value):
		self._undisturbedMediumFlow = value

	@property
	def pressureValveInlet(self):
		return self._pressureValveInlet

	@pressureValveInlet.setter
	def pressureValveInlet(self, value):
		self._pressureValveInlet = value 

	@property
	def pressureValveOutlet(self):
		return self._pressureValveOutlet

	@pressureValveOutlet.setter
	def pressureValveOutlet(self, value):
		self._pressureValveOutlet = value  

	@property
	def mediumTemperature(self):
		return self._mediumTemperature

	@mediumTemperature.setter
	def mediumTemperature(self, value):
		self._mediumTemperature = value

	@property
	def rodDisplacement(self):
		return self._rodDisplacement

	@rodDisplacement.setter
	def rodDisplacement(self, value):
		self._rodDisplacement = value

	@property
	def disturbedMediumFlow(self):
		return self._disturbedMediumFlow

	@disturbedMediumFlow.setter
	def disturbedMediumFlow(self, value):
		self._disturbedMediumFlow = value 

	@property
	def selectedFault(self):
		return self._selectedFault

	@selectedFault.setter
	def selectedFault(self, value):
		self._selectedFault = value 

	@property
	def faultType(self):
		return self._faultType

	@faultType.setter
	def faultType(self, value):
		self._faultType = value

	@property
	def faultIntensity(self):
		return self._faultIntensity

	@faultIntensity.setter
	def faultIntensity(self, value):
		self._faultIntensity = value                 

	def __str__(self):
		return "<ValveReading(elementId = '%s', timestamp = '%s', externalControllerOutput = '%s', undisturbedMediumFlow = '%s', pressureValveInlet = '%s',\
		 pressureValveOutlet = '%s', mediumTemperature = '%s', rodDisplacement = '%s', disturbedMediumFlow = '%s', selectedFault = '%s', faultType = '%s', faultIntensity = '%s')>" \
		% (self._id, self._timestamp, self._externalControllerOutput, self._undisturbedMediumFlow, self._pressureValveInlet, self._pressureValveOutlet, self._mediumTemperature,\
		 self._rodDisplacement, self._disturbedMediumFlow, self._selectedFault, self._faultType, self._faultIntensity)


