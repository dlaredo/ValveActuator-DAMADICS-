#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mysql.h>

int connect_to_DB(void);
int writeToDB(time_t);
void updateSensorValues(double);
int initValues(void);

typedef struct
{
	double controlValue;
	double pressureValveInlet;
	double pressureValveOutlet;
	double rodDisplacement;
	double disturbedMediumFlow;
	double mediumTemperature;
	double faultIntensity;
	int selectedFault;
	int faultType;
} damadicsSensorValues;

MYSQL *con;
damadicsSensorValues sensorValues;
int alreadySampled;
time_t startDateTime;
FILE *startDateTimeFile;
unsigned long lastSampleTime;

int main(int argc, char **argv)
{

	int sampleTime = 60, writingStatus = 0;
	double currentTime = 0, timeDelta = 0.0025; //Current time
	unsigned long intTime = 0;

	if(initValues() != 0)
	{
		fprintf(stdout, "%s\n", "Initialization error");
		return 0;
	}

	while(currentTime < 250)
	{
		intTime = (unsigned long)floor(currentTime);

		//fprintf(stdout, "currentTime %lf int time %li frac time %lf intTimeDummy %lf\n", currentTime, intTime);

		//Take only one sample per sample cycle (minimum sample rate is 1 second due to precision issues)
		if(intTime%sampleTime == 0 && lastSampleTime != intTime)
		{
				lastSampleTime = intTime;
				updateSensorValues(intTime);
				writingStatus = writeToDB((time_t)intTime);

				if(writingStatus != 0)
				{
					fprintf(stdout, "%s\n", "Error ocurred while writing to the DB");
					return 0;
				}
		}

		currentTime += timeDelta;
		//fprintf(stdout, "Current Time: %lf\n", currentTime);
	}

	fclose(startDateTimeFile);
	mysql_close(con);
	mysql_library_end();

  	return 0;
}

int initValues(void)
{
	int bytesRead = 0, connectionStatus = 0;

	lastSampleTime = -1; //Initilize to -1 to take the sample at time 0
	sensorValues.controlValue = 0;
	sensorValues.pressureValveInlet = 0;
	sensorValues.pressureValveOutlet = 0;
	sensorValues.rodDisplacement = 0;
	sensorValues.disturbedMediumFlow = 0;
	sensorValues.mediumTemperature = 0;
	sensorValues.faultIntensity = 0;
	sensorValues.selectedFault = 0;
	sensorValues.faultType = 0;

	if((startDateTimeFile = fopen("lastDateTime.txt", "r+")) != NULL)
	{
		bytesRead = fscanf(startDateTimeFile, "%li", &startDateTime);

		if(bytesRead == 0 || bytesRead == EOF)
			startDateTime = time(NULL);
	}
	else
	{
		if((startDateTimeFile = fopen("lastDateTime.txt", "w+")) == NULL)
			return -1;

		startDateTime = time(NULL);
	}

	connectionStatus = connect_to_DB();

	if(connectionStatus == 0)
  		fprintf(stdout, "Succesfully connected to: %s\n", mysql_get_client_info());
  	else
  	{
  		fprintf(stdout, "Connection failed");
  		return -1;
  	}

	return 0;
}

int connect_to_DB(void)
{

	printf("Before mysql_init\n");
  con = mysql_init(NULL);
  printf("after mysql_init\n");

  if (con == NULL) 
  {
      fprintf(stdout, "%s\n", mysql_error(con));
      return -1;
  }

  if (mysql_real_connect(con, "192.168.56.1", "controlslab", "controlslab", "damadics2", 3306, NULL, 0) == NULL) 
  {
      fprintf(stdout, "%s\n", mysql_error(con));
      mysql_close(con);
      return -1;
  }

  return 0;

}

int writeToDB(time_t elapsedSeconds)
{

	char queryString[500], dateTimeStr[100];
	time_t currentSimulationTime = startDateTime + elapsedSeconds;
	struct tm simulationDateTime = *localtime(&currentSimulationTime);

	sprintf(dateTimeStr, "%04d-%02d-%02d %02d:%02d:%02d", simulationDateTime.tm_year + 1900, simulationDateTime.tm_mon + 1, 
		simulationDateTime.tm_mday, simulationDateTime.tm_hour, simulationDateTime.tm_min, simulationDateTime.tm_sec);

	sprintf(queryString, "INSERT INTO valveReadings(timestamp, externalControllerOutput, disturbedMediumFlow,\
		pressureValveInlet, pressureValveOutlet, mediumTemperature, rodDisplacement, selectedFault, faultType, faultIntensity)\
		VALUES ('%s', %f, %f, %f, %f, %f, %f, %d, %d, %f)", dateTimeStr, sensorValues.controlValue, sensorValues.disturbedMediumFlow, 
		sensorValues.pressureValveInlet, sensorValues.pressureValveOutlet, sensorValues.mediumTemperature, sensorValues.rodDisplacement, 
		sensorValues.selectedFault, sensorValues.faultType, sensorValues.faultIntensity);

	fprintf(stdout, "%s\n", queryString);

	if (mysql_query(con, queryString))
  	{
  		fprintf(stdout, "%s\n", mysql_error(con));
  		return -1;
  	}
  	else
  	{
  		fprintf(stdout, "Writing to DB at: %li\n", currentSimulationTime);
  		fprintf(stdout, "fseek result %d\n", fseek(startDateTimeFile, 0, SEEK_SET));
  		fprintf(startDateTimeFile, "%li\n", currentSimulationTime+1);
  	}

  	return 0;
}

void updateSensorValues(double dummy)
{
	sensorValues.controlValue = dummy;
	sensorValues.pressureValveInlet = dummy+1;
	sensorValues.pressureValveOutlet = dummy+2;
	sensorValues.rodDisplacement = dummy+3;
	sensorValues.disturbedMediumFlow = dummy+4;
	sensorValues.mediumTemperature = dummy+5;
	sensorValues.faultIntensity = dummy+6;
	sensorValues.selectedFault = 20;
	sensorValues.faultType = 4;
}

