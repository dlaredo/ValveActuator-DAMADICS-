/*
 * sfuntmpl_basic.c: Basic 'C' template for a level 2 S-function.
 *
 * Copyright 1990-2013 The MathWorks, Inc.
 */


/*
 * You must specify the S_FUNCTION_NAME as the name of your S-function
 * (i.e. replace sfuntmpl_basic with the name of your S-function).
 */

#define S_FUNCTION_NAME  saveToDB
#define S_FUNCTION_LEVEL 2

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#include "simstruc.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mysql.h>
#include "utils.h"

#define INPUT_PARAMS 1
#define BULK_INSERT_SIZE 100
#define STRING_APPROX_SIZE 150

int connect_to_DB(char *, char *, char *, char *);
int writeToDB(time_t);
void updateSensorValues(const real_T *);
int initValues(SimStruct *S);
int addToBuffer(time_t);
int dbBulkInsert(void);
void cleanBuffer(void);

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

//Ceate a buffer of elements to be inserted in the DB
typedef struct
{
  time_t timestamp; 
  damadicsSensorValues sensorValues;
} databaseEntry;

MYSQL *con;
damadicsSensorValues sensorValues;
databaseEntry databaseElement[BULK_INSERT_SIZE]; 
int alreadySampled, bufferCounter;
time_t startDateTime;
FILE *startDateTimeFile, *logFile;
unsigned long lastSampleTime;
int stopFlag, connectionUp;


/* Error handling
 * --------------
 *
 * You should use the following technique to report errors encountered within
 * an S-function:
 *
 *       ssSetErrorStatus(S,"Error encountered due to ...");
 *       return;
 *
 * Note that the 2nd argument to ssSetErrorStatus must be persistent memory.
 * It cannot be a local variable. For example the following will cause
 * unpredictable errors:
 *
 *      mdlOutputs()
 *      {
 *         char msg[256];         {ILLEGAL: to fix use "static char msg[256];"}
 *         sprintf(msg,"Error due to %s", string);
 *         ssSetErrorStatus(S,msg);
 *         return;
 *      }
 *
 */

/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, INPUT_PARAMS);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /* Return if number of expected != number of actual parameters */
        return;
    }

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 1)) return;
    ssSetInputPortWidth(S, 0, DYNAMICALLY_SIZED);
    ssSetInputPortRequiredContiguous(S, 0, true); /*direct input signal access*/
    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     */
    ssSetInputPortDirectFeedThrough(S, 0, 1);

    /*7 output signals, all of them scalar*/
    if (!ssSetNumOutputPorts(S, 0)) return;

    ssSetNumSampleTimes(S, 1); //This is extremely important as it may affect the probability of a fault being drawn
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

    ssSetSFcnParamTunable(S,0,false);

    /* Specify the sim state compliance to be same as a built-in block */
    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);

    ssSetOptions(S, 0);
}



/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    This function is used to specify the sample time(s) for your
 *    S-function. You must register the same number of sample times as
 *    specified in ssSetNumSampleTimes.
 */
//This is extremely important as it may affect the probability of a fault being drawn
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, INHERITED_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);

}



#define MDL_INITIALIZE_CONDITIONS   /* Change to #undef to remove function */
#if defined(MDL_INITIALIZE_CONDITIONS)
  /* Function: mdlInitializeConditions ========================================
   * Abstract:
   *    In this function, you should initialize the continuous and discrete
   *    states for your S-function block.  The initial states are placed
   *    in the state vector, ssGetContStates(S) or ssGetRealDiscStates(S).
   *    You can also perform any other initialization activities that your
   *    S-function may require. Note, this routine will be called at the
   *    start of simulation and if it is present in an enabled subsystem
   *    configured to reset states, it will be call when the enabled subsystem
   *    restarts execution to reset the states.
   */
  static void mdlInitializeConditions(SimStruct *S)
  {
    if(initValues(S) != 0)
    {
      ssPrintf("Initialization error\n");
      stopFlag = 1;
      ssSetStopRequested(S, 1);
    }
    ssPrintf("Variables Initialized in saveToDB. Rest will be written to the log.\n");
  }
#endif /* MDL_INITIALIZE_CONDITIONS */



#undef MDL_START  /* Change to #undef to remove function */
#if defined(MDL_START) 
  /* Function: mdlStart =======================================================
   * Abstract:
   *    This function is called once at start of model execution. If you
   *    have states that should be initialized once, this is the place
   *    to do it.
   */
  static void mdlStart(SimStruct *S)
  {
  }
#endif /*  MDL_START */



/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block.
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    const real_T *signalVector = (const real_T*) ssGetInputPortSignal(S,0);
    double clockTime = 0;
    int sampleTime = 60, status = 0;
    unsigned long intTime = 0;

    clockTime = signalVector[9];

    intTime = (unsigned long)floor(clockTime);

    if(stopFlag != 1) //Try to write to database only if the stop flag is not raised
    {

      //Take only one sample per sample cycle (minimum sample rate is 1 second due to precision issues)
      if(intTime%sampleTime == 0 && lastSampleTime != intTime)
      {
          lastSampleTime = intTime;
          updateSensorValues(signalVector);
          //writingStatus = writeToDB((time_t)intTime);
          status = addToBuffer((time_t)intTime);

          if(status != 0)
          {
            logMsg(logFile, "Error ocurred while writing to the DB. Halting simulation.\n");
            sendMail("Error ocurred while writing to the DB. Halting simulation.\n");
            ssSetStopRequested(S, 1); //Stop simulation if writing to the DB is not possible
          }
      }
    }
    else
      ssSetStopRequested(S, 1);
}



#undef MDL_UPDATE  /* Change to #undef to remove function */
#if defined(MDL_UPDATE)
  /* Function: mdlUpdate ======================================================
   * Abstract:
   *    This function is called once for every major integration time step.
   *    Discrete states are typically updated here, but this function is useful
   *    for performing any tasks that should only take place once per
   *    integration step.
   */
  static void mdlUpdate(SimStruct *S, int_T tid)
  {
  }
#endif /* MDL_UPDATE */



#undef MDL_DERIVATIVES  /* Change to #undef to remove function */
#if defined(MDL_DERIVATIVES)
  /* Function: mdlDerivatives =================================================
   * Abstract:
   *    In this function, you compute the S-function block's derivatives.
   *    The derivatives are placed in the derivative vector, ssGetdX(S).
   */
  static void mdlDerivatives(SimStruct *S)
  {
  }
#endif /* MDL_DERIVATIVES */



/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was
 *    allocated in mdlStart, this is the place to free it.
 */
static void mdlTerminate(SimStruct *S)
{
  if(connectionUp == 1)
    mysql_close(con);
  else
    logMsg(logFile, "No DB connection");
  
  fclose(startDateTimeFile);
  fclose(logFile);
}


/*Function specific functions (implementation)*/

int initValues(SimStruct *S)
{
  int bytesRead = 0, connectionStatus = 0;
  char host[32]={}, user[128]={}, pass[128]={}, db[16]={}, paramString[512];

  lastSampleTime = -1; //Initilize to -1 to take the sample at time 0
  stopFlag = 0; //Set stop flag to 0
  sensorValues.controlValue = 0;
  sensorValues.pressureValveInlet = 0;
  sensorValues.pressureValveOutlet = 0;
  sensorValues.rodDisplacement = 0;
  sensorValues.disturbedMediumFlow = 0;
  sensorValues.mediumTemperature = 0;
  sensorValues.faultIntensity = 0;
  sensorValues.selectedFault = 0;
  sensorValues.faultType = 0;

  char msg[128];
  connectionUp = 0;
  bufferCounter = 0;

  logFile = fopen("DamadicsDatabaseLog.txt", "a");

  if(logFile == NULL){
    ssPrintf("Could not open log file for Database\n");
    return -1;
  }

  //Read connection parameters for the database
  if(mxGetString(ssGetSFcnParam(S, 0), paramString, mxGetN(ssGetSFcnParam(S, 0))*sizeof(mxChar) + 1) != 0){
    logMsg(logFile, "Unable to read parameters for database connection\n");
    return -1;
  }

  dbParamsFromString(paramString, host, user, pass, db);

  //sprintf(msg, "DB Params:%s\n host:%s user:%s pass:%s db:%s", paramString, host, user, pass, db);
  //logMsg(logFile, msg);

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

  connectionStatus = connect_to_DB(host, user, pass, db);

  if(connectionStatus == 0)
  {
    sprintf(msg, "Succesfully connected to: %s\n", (char *)mysql_get_client_info());
    logMsg(logFile, msg);
    connectionUp = 1;
  }
  else
  {
    logMsg(logFile, "Connection to the database failed\n");
    return -1;
  }

  return 0;
}

int connect_to_DB(char *host, char *user, char *pass, char *db)
{

  con = mysql_init(NULL);
  char msg[350];

  if (con == NULL) 
  {
      sprintf(msg, "%s\n", mysql_error(con));
      logMsg(logFile, msg);
      return -1;
  }
  
  //sprintf(msg, "Attempting connection to DB with Params: host:%s user:%s pass:%s db:%s", host, user, pass, db);
  //logMsg(logFile, msg);
  
  if (mysql_real_connect(con, host, user, pass, db, 3306, NULL, 0) == NULL) 
  {
      sprintf(msg, "Unable to connect to DB\n%s\n", mysql_error(con));
      logMsg(logFile, msg);
      //mysql_close(con);
      return -1;
  }  

  return 0;

}

int writeToDB(time_t elapsedSeconds)
{

  char queryString[500], dateTimeStr[100], msg[128];
  time_t currentSimulationTime = startDateTime + elapsedSeconds;
  struct tm simulationDateTime = *localtime(&currentSimulationTime);

  sprintf(dateTimeStr, "%04d-%02d-%02d %02d:%02d:%02d", simulationDateTime.tm_year + 1900, simulationDateTime.tm_mon + 1, 
    simulationDateTime.tm_mday, simulationDateTime.tm_hour, simulationDateTime.tm_min, simulationDateTime.tm_sec);

  sprintf(queryString, "INSERT INTO valveReadings(timestamp, externalControllerOutput, disturbedMediumFlow,\
    pressureValveInlet, pressureValveOutlet, mediumTemperature, rodDisplacement, selectedFault, faultType, faultIntensity)\
    VALUES ('%s', %f, %f, %f, %f, %f, %f, %d, %d, %f)", dateTimeStr, sensorValues.controlValue, sensorValues.disturbedMediumFlow, 
    sensorValues.pressureValveInlet, sensorValues.pressureValveOutlet, sensorValues.mediumTemperature, sensorValues.rodDisplacement, 
    sensorValues.selectedFault, sensorValues.faultType, sensorValues.faultIntensity);

  if (mysql_query(con, queryString))
  {
    sprintf(msg, "%s\n", mysql_error(con));
    logMsg(logFile, msg);
    return -1;
  }
  else
  { //Need to improve the performance of this. Not to write the simulation time at each iteration
    fseek(startDateTimeFile, 0, SEEK_SET);
    fprintf(startDateTimeFile, "%li\n", currentSimulationTime+1);
  }

    return 0;
}

int dbBulkInsert()
{

  int i = 0;
  char insertString[256] = {}, dateTimeStr[24] = {}, msg[128] = {}; 
  char valueBulkStr[BULK_INSERT_SIZE*STRING_APPROX_SIZE+256] = {}, valueStr[STRING_APPROX_SIZE] = {};
  damadicsSensorValues sValues;
  time_t elementTimestamp;
  struct tm elementDateTime; 

  FILE *fp = NULL;
  fp = fopen("lastBulkInsertLog.txt", "w");

  sprintf(insertString, "INSERT INTO valveReadings(timestamp, externalControllerOutput, disturbedMediumFlow,\
    pressureValveInlet, pressureValveOutlet, mediumTemperature, rodDisplacement, selectedFault, faultType, faultIntensity)\
    VALUES ");

  strcpy(valueBulkStr, insertString);

  for(i = 0; i < BULK_INSERT_SIZE; i++)
  {
    elementTimestamp = databaseElement[i].timestamp;
    elementDateTime = *localtime(&elementTimestamp);
    sValues = databaseElement[i].sensorValues;

    sprintf(dateTimeStr, "%04d-%02d-%02d %02d:%02d:%02d", elementDateTime.tm_year + 1900, elementDateTime.tm_mon + 1, 
    elementDateTime.tm_mday, elementDateTime.tm_hour, elementDateTime.tm_min, elementDateTime.tm_sec);


    if(i != (BULK_INSERT_SIZE - 1))
    {
      sprintf(valueStr, "('%s', %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %.6f), ", dateTimeStr, sValues.controlValue, 
      sValues.disturbedMediumFlow, sValues.pressureValveInlet, sValues.pressureValveOutlet, sValues.mediumTemperature, 
      sValues.rodDisplacement, sValues.selectedFault, sValues.faultType, sValues.faultIntensity);
    }
    else //For the last element omit the comma at the end of the string
    {
      sprintf(valueStr, "('%s', %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %d, %d, %.6f)", dateTimeStr, sValues.controlValue, 
      sValues.disturbedMediumFlow, sValues.pressureValveInlet, sValues.pressureValveOutlet, sValues.mediumTemperature, 
      sValues.rodDisplacement, sValues.selectedFault, sValues.faultType, sValues.faultIntensity);
    }

    strcat(valueBulkStr, valueStr);
  }

  fprintf(fp, "%s\n", valueBulkStr);
  fclose(fp);

  if (mysql_query(con, valueBulkStr))
  {
    sprintf(msg, "%s\n", mysql_error(con));
    logMsg(logFile, msg);
    return -1;
  }
  else
  {
    fseek(startDateTimeFile, 0, SEEK_SET);
    fprintf(startDateTimeFile, "%li\n", elementTimestamp+1);
  }

  return 0;
}

void updateSensorValues(const real_T *signalVector)
{
  sensorValues.controlValue = (double)signalVector[0];
  sensorValues.pressureValveInlet = (double)signalVector[1];
  sensorValues.pressureValveOutlet = (double)signalVector[2];
  sensorValues.rodDisplacement = (double)signalVector[3];
  sensorValues.disturbedMediumFlow = (double)signalVector[4];
  sensorValues.mediumTemperature = (double)signalVector[5];
  sensorValues.faultIntensity = (double)signalVector[6];
  sensorValues.selectedFault = (int)signalVector[7];
  sensorValues.faultType = (int)signalVector[8];
}

int addToBuffer(time_t elapsedSeconds)
{
  time_t currentSimulationTime = startDateTime + elapsedSeconds;

  if(bufferCounter < BULK_INSERT_SIZE)
  {
    databaseElement[bufferCounter].timestamp = currentSimulationTime;
    databaseElement[bufferCounter].sensorValues = sensorValues;
    bufferCounter += 1;
  }
  else
  {
    if(dbBulkInsert() == 0) //Attempt the bulk insert to mysql
    {
      cleanBuffer();
      bufferCounter = 0;
    }
    else
      return -1;
  }  

  return 0;
}

void cleanBuffer()
{
  int i = 0;

  for(i = 0; i < BULK_INSERT_SIZE; i++)
  {
    databaseElement[i].timestamp = 0;
    databaseElement[i].sensorValues.controlValue = 0;
    databaseElement[i].sensorValues.pressureValveInlet = 0;
    databaseElement[i].sensorValues.pressureValveOutlet = 0;
    databaseElement[i].sensorValues.rodDisplacement = 0;
    databaseElement[i].sensorValues.disturbedMediumFlow = 0;
    databaseElement[i].sensorValues.mediumTemperature = 0;
    databaseElement[i].sensorValues.faultIntensity = 0;
    databaseElement[i].sensorValues.selectedFault = 0;
    databaseElement[i].sensorValues.faultType = 0;
  }
}

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
