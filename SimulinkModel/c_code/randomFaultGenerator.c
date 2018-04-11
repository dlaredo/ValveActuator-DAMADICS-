/*
 * sfuntmpl_basic.c: Basic 'C' template for a level 2 S-function.
 *
 * Copyright 1990-2013 The MathWorks, Inc.
 */


/*
 * You must specify the S_FUNCTION_NAME as the name of your S-function
 * (i.e. replace sfuntmpl_basic with the name of your S-function).
 */

#define S_FUNCTION_NAME  randomFaultGenerator
#define S_FUNCTION_LEVEL 2

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 */
#include "simstruc.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utils.h"

#define INPUT_PARAMS 2

void initValues(SimStruct *);
void setNoFaultParameters(void);
void randomFaultGenerator(double);
int getBooleanFromRandom(float, float, float);
void getRandomFault(int *, int *);
void setSimulationParameters(int, int, double, double);
double gaussrand(void);
int faultMap(int);

typedef struct
{
  double FDS;
  double To;
  int FSD;
  int Fsel;
  int Ftype;
  double From;
  double FMS;
} signalParameters;

/*Specify how many faults and what types there are for each fault*/
int fmap[4];

/*Global model variables*/
int faultInProcess, simulationRunning, generatingFault;
double timeLimit, faultStartTime; //Using double for the times given the precision issues with float
FILE *logFile;

//Horizon for each of the faults
double horSeconds[4] = {1500, 2400, 5100, 85500};

//Structure for the parameters of the fault
signalParameters sParams;

//Parameters for the simulation of the faul (treshgen and treshstop)
float treshGen, treshStop;


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
    if (!ssSetNumOutputPorts(S, 7)) return;
    ssSetOutputPortWidth(S, 0, 1); 
    ssSetOutputPortWidth(S, 1, 1);
    ssSetOutputPortWidth(S, 2, 1);
    ssSetOutputPortWidth(S, 3, 1);
    ssSetOutputPortWidth(S, 4, 1);
    ssSetOutputPortWidth(S, 5, 1);
    ssSetOutputPortWidth(S, 6, 1);

    ssSetNumSampleTimes(S, 1); //This is extremely important as it may affect the probability of a fault being drawn
    ssSetNumRWork(S, 0);
    ssSetNumIWork(S, 0);
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);

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
    initValues(S);
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
    const real_T *currentTime = (const real_T*) ssGetInputPortSignal(S,0);
    real_T       *fsel = ssGetOutputPortSignal(S,0);
    real_T       *ftype = ssGetOutputPortSignal(S,1);
    real_T       *from = ssGetOutputPortSignal(S,2);
    real_T       *to = ssGetOutputPortSignal(S,3);
    real_T       *fds = ssGetOutputPortSignal(S,4);
    real_T       *fsd = ssGetOutputPortSignal(S,5);
    real_T       *fms = ssGetOutputPortSignal(S,6);
    
    //Generate a random fault
    randomFaultGenerator((double)*currentTime);

    fsel[0] = (real_T)sParams.Fsel;
    ftype[0] = (real_T)sParams.Ftype;
    from[0] = (real_T)sParams.From;
    to[0] = (real_T)sParams.To;
    fds[0] = (real_T)sParams.FDS;
    fsd[0] = (real_T)sParams.FSD;
    fms[0] = (real_T)sParams.FMS;
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
  logMsg(logFile, "Simulation Terminated\n");
  fclose(logFile);
}


/*Function specific functions (implementation)*/

void initValues(SimStruct *S)
{
  faultStartTime = 0;
  timeLimit = 2592000; //Equivalent to 30 days of simulation
  logFile = fopen("rFaultGeneratorLog.txt", "a");
  faultInProcess = 0;
  simulationRunning = 1;
  generatingFault = 0;

  treshGen = *mxGetPr(ssGetSFcnParam(S, 0));
  treshStop = *mxGetPr(ssGetSFcnParam(S, 1));

  char simParams[128] = {};
  sprintf(simParams, "Simulation started with threshGen: %f and treshStop %f", treshGen, treshStop);
  logMsg(logFile, simParams);

  setNoFaultParameters();

  if(logFile == NULL)
    ssPrintf("Could not open log file for randomFaultGenerator\n");

  ssPrintf("Variables Initialized in randomFaultGenerator. Rest will be written to the log. \n");
}

void setNoFaultParameters(void)
{
  //Initialize to fault parameters
  sParams.Fsel = 20;
  sParams.Ftype = 1;
  sParams.FDS = 0;
  sParams.To = timeLimit;
  sParams.FSD = 1;
  sParams.From = 0;
  sParams.FMS = 0;
}

void randomFaultGenerator(double currentTime)
{
  float muGen = 0, stdGen = 1;
  float muStop = 0, stdStop = 1;
  double offsetTime = 0, elapsedSeconds = 0;
  int b = 0, Fsel = 0, Ftype = 0, horizonSeconds = 0;
  char msg[350];

  //One fault at the time only
  if(faultInProcess == 0 && generatingFault == 0)
  {
    b = getBooleanFromRandom(muGen, stdGen, treshGen);

    /*If no fault in process and generateRandomFault indicates that a
        fault should be generated*/

        if(b == 1)
        {

          generatingFault = 1; //Used for safety to avoid creating more than one fault
          getRandomFault(&Fsel, &Ftype);
          setSimulationParameters(Fsel, Ftype, currentTime, offsetTime);
          
          if(sParams.Fsel != 20) //If the parameters for the fault could be set, then
          {
            faultInProcess = 1;
            faultStartTime = currentTime;
            sprintf(msg, "\n\nGenerating Fault: %d of Type:%d at time %lf for at least %lf seconds\nSimulation parameters: From: %f, FDS:%f, To:%lf, FSD:%d\n", 
             sParams.Fsel, sParams.Ftype, currentTime, horSeconds[sParams.Ftype-1], sParams.From, sParams.FDS, sParams.To, sParams.FSD);
            logMsg(logFile, msg);
          }

          generatingFault = 0;

        }
  }
  else
  {
    //If there is a fault in process, see if its time to stop it

    elapsedSeconds = currentTime - faultStartTime;
    Ftype = sParams.Ftype;

    //Only stop if the fault has completely developed according to the fault type
    horizonSeconds = horSeconds[Ftype-1];

    if(elapsedSeconds > horizonSeconds)
    {
      b = getBooleanFromRandom(muStop, stdStop, treshStop);

      if(b == 1)
      {
        sprintf(msg, "Stopping Fault: %d of Type:%d at time %lf after %lf seconds\n", sParams.Fsel, sParams.Ftype, currentTime, elapsedSeconds);
        logMsg(logFile, msg);
        setNoFaultParameters();
        faultInProcess = 0;
      }
    }
  }
}

/*generate a flag using a randomly distributed number*/
int getBooleanFromRandom(float mu, float std, float tresh)
{
  float r;

  r = fabs(gaussrand());
  r = r*std+mu;

  if(r > tresh)
    return 1;

  return 0;
}

/*Get a random fault with a random type*/
void getRandomFault(int *fsel, int *Ftype)
{

  int rFtypeIndex = 0, nFaultTypes = 0;

  *fsel = rand()%19 + 1;
  nFaultTypes = faultMap(*fsel);
  rFtypeIndex = rand()%nFaultTypes;

  *Ftype = fmap[rFtypeIndex];
}

/*Set the parameters for the simulation of a specified fault. Taken from DGen mask init*/
void setSimulationParameters(int Fsel, int Ftype, double currentTime, double offsetTime)
{

  double MFS = 0, FDT = 0;
  char msg[350];

  if(Ftype == 4)//Abrupt fault
  {
    MFS = 1;

    if(Fsel == 2 || Fsel == 3 || Fsel == 5 || Fsel == 6 || Fsel == 9 || Fsel == 11 || Fsel == 18)
      FDT = 3600*24; //1 day to develop the fault
    else if(Fsel == 4 || Fsel == 17)
      FDT = 3600; //1 Hour to develop the fault
    else if(Fsel == 13)
      FDT = 60*15; // 15 minutes to develop the fault
    else
    {
      sprintf(msg, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
      logMsg(logFile, msg);
      Fsel = 20; //No fault selected
      FDT = 0;
    }
  }
  else
  {
    MFS = 0.25*Ftype;
    FDT = 0;

    if(Fsel == 3 || Fsel == 4 || Fsel == 5 || Fsel == 6 || Fsel == 9)
    {
      sprintf(msg, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
      logMsg(logFile, msg);
      Fsel = 20; //No fault selected
    }
    else if(Fsel == 2 || Fsel == 11 || Fsel == 15 || Fsel == 17)
    {
      if(Ftype < 3)
      {
        sprintf(msg, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
        logMsg(logFile, msg);
        Fsel = 20; //No fault selected
      }
    }
  }

  sParams.From = currentTime +  offsetTime; /*evalin('base', 'From');*/ 
  sParams.Fsel = Fsel;
  sParams.Ftype = Ftype;
  sParams.FMS = MFS;

  if(Fsel == 12 || Fsel == 14 || Fsel == 19)
    sParams.FSD = -1;

  if(FDT == 0)
    sParams.FDS = 999999999999999; //Very long number to simulate an inf slope
  else
    sParams.FDS = MFS/FDT;
}

//Generate Gaussian distributed numbers with mean 0 and std 1 according to Knuth
/*To adjust to some other distribution, multiply by the standard deviation and add the mean.*/
double gaussrand(void)
{
  static double V1, V2, S;
  static int phase = 0;
  double X;

  if(phase == 0) {
    do {
      double U1 = (double)rand() / RAND_MAX;
      double U2 = (double)rand() / RAND_MAX;

      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
      } while(S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);
  } else
    X = V2 * sqrt(-2 * log(S) / S);

  phase = 1 - phase;

  return X;
}

/*Specify how many faults and what types there are for each fault*/
int faultMap(int Fsel)
{

  int nFaultTypes = 0;

  switch(Fsel)
  {
    case 1:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 2:
    {
      fmap[0] = 3; fmap[1] = 4;
      nFaultTypes = 2;
      break;
    }
    case 3:
    {
      fmap[0] = 4;
      nFaultTypes = 1;
      break;
    }
    case 4:
    {
      fmap[0] = 4;
      nFaultTypes = 1;
      break;
    }
    case 5:
    {
      fmap[0] = 4;
      nFaultTypes = 1;
      break;
    }
    case 6:
    {
      fmap[0] = 4;
      nFaultTypes = 1;
      break;
    }
    case 7:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 8:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 9:
    {
      fmap[0] = 4;
      nFaultTypes = 1;
      break;
    }
    case 10:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 11:
    {
      fmap[0] = 3; fmap[1] = 4;
      nFaultTypes = 2;
      break;
    }
    case 12:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 13:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3; fmap[3] = 4;
      nFaultTypes = 4;
      break;
    }
    case 14:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 15:
    {
      fmap[0] = 3;
      nFaultTypes = 1;
      break;
    }
    case 16:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
    case 17:
    {
      fmap[0] = 3; fmap[1] = 4;
      nFaultTypes = 2;
      break;
    }
    case 18:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3; fmap[3] = 4;
      nFaultTypes = 4;
      break;
    }
    case 19:
    {
      fmap[0] = 1; fmap[1] = 2; fmap[2] = 3;
      nFaultTypes = 3;
      break;
    }
  }

  return nFaultTypes;
}

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
