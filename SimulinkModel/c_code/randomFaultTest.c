#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void initValues(void);
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
	int DSim;
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


int main(int argc, char *argv[])
{
	double currentTime = 0, delta = 0.0025;

	initValues();

	srand(time(0)); //use current time as seed for random generator

	while(1)
	{
		randomFaultGenerator(currentTime);
		currentTime += delta;
	}

	return 0;
}

void initValues(void)
{
	faultStartTime = 0;
	timeLimit = 2592000; //Equivalent to 30 days of simulation
	logFile = fopen("rFaultGeneratorLog.txt", "a");
	faultInProcess = 0;
	simulationRunning = 0;
	generatingFault = 0;

	setNoFaultParameters();
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
	sParams.DSim = 0;
}

void randomFaultGenerator(double currentTime)
{
	float muGen = 0, stdGen = 1, treshGen = 5.7;
	float muStop = 0, stdStop = 1, treshStop = 2;
	double offsetTime = 0, elapsedSeconds = 0;
	int b = 0, Fsel = 0, Ftype = 0, horizonSeconds = 0;

	if(currentTime > timeLimit)
	{
		simulationRunning = 0;
		sParams.DSim = 1;

		fprintf(logFile, "Time up\n");
		fclose(logFile);
		exit(0);
	}

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
        		fprintf(logFile, "\n\nGenerating Fault: %d of Type:%d at time %lf for at least %lf seconds\n", sParams.Fsel, sParams.Ftype, currentTime, horSeconds[sParams.Ftype-1]);
        		fprintf(logFile, "Simulation parameters: From: %f, FDS:%f, To:%lf, FSD:%d\n", sParams.From, sParams.FDS, sParams.To, sParams.FSD);
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
				fprintf(logFile, "Stopping Fault: %d of Type:%d at time %lf after %lf seconds\n", sParams.Fsel, sParams.Ftype, currentTime, elapsedSeconds);
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
			fprintf(logFile, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
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
			fprintf(logFile, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
			Fsel = 20; //No fault selected
		}
		else if(Fsel == 2 || Fsel == 11 || Fsel == 15 || Fsel == 17)
		{
			if(Ftype < 3)
			{
				fprintf(logFile, "Error! - Fault %d with type %d not specified for benchmark purpose\n", Fsel, Ftype);
				Fsel = 20; //No fault selected
			}
		}
	}

	sParams.From = currentTime +  offsetTime; /*evalin('base', 'From');*/ 
	sParams.Fsel = Fsel;
	sParams.Ftype = Ftype;

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







