#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct
{
	double FDS;
	long To;
	int FSD;
	int Fsel;
	float From;
	int DSim;
} signalParameters;

/*Specify how many faults and what types there are for each fault*/
int fmap[4];


void main(int argc, char *argv[])
{

	srand(time(0)); //use current time as seed for random generator
}

void randomFaultGenerator(float currentTime)
{
	float muGen = 0, stdGen = 0, treshGen = 5.7;
	float muStop = 0, stdStop = 1, treshStop = 2;
	float offsetTime = 1;

	//logFile = evalin('base', 'logFileDescriptor');
	FILE *logFile = NULL;

	int horSeconds[4] = {1500, 2400, 5100, 85500};
	int Fsel = 0, ftype = 0, faultInProcess = 0, timeLimit = 0
	int DSim = 0;

	/*
	FSel = evalin('base', 'FSel');
    ftype = evalin('base', 'ftype');
    faultInProcess = evalin('base', 'faultInProcess');
    timeLimit = evalin('base', 'timeLimit');
    */

	if(currentTime > timeLimit)
	{
		simulationRunning = 0;
		//assignin('base', 'simulationRunning', 0);
		//dbConn = evalin('base', 'dbconn');
		Dsim = 1;

		fclose(logFile);
		//close(dbConn)
		//set_param(strcat(DGenBlockAddress, '/Disable simulation'), 'value', num2str(DSim));
	}

	//One fault at the time only
	if(faultInProcess == 0)
	{
		b = generateRandomFault(muGen, stdGen, treshGen);

		/*If no fault in process and generateRandomFault indicates that a
        fault should be generated*/

        if(b == 1)
        {

        	faultInProcess = 1;

        }
	}
}

/*generate a flag using a randomly distributed number*/
int getBooleanFromRandom(mu, std, tresh)
{
	float r;

	r = gaussrand();
	r = r*std+mu;

	if(r > tresh)
		return 1;

	return 0;
}

/*Get a random fault with a random type*/
void getRandomFault(int *fsel, int *ftype)
{

	int rftype = 0, nFaultTypes = 0;

	*fsel = rand()%19 + 1;
	nFaultTypes = faultMap(*fsel);
	rftype = rand()%nFaultTypes;
	*ftype = faultMap[rftype];
}

/*Set the parameters for the simulation of a specified fault. Taken from DGen mask init*/
signalParameters getSimulationParameters(int FSel, int FType, float currentTime, float offsetTime)
{
	signalParameters sParams;

	float MFS = 0, FDT = 0;

	sParams.DSim = 0;
	sParams.FSD = 1;
	sParams.From = currentTime +  offsetTime; /*evalin('base', 'From');*/ 
	sParams.To = 99999999999999999; //Long number
	sParams.Fsel = Fsel;
	sParams.FDS = 0;

	if(FType == 4)//Abrupt fault
	{
		MFS = 1;

		if(FSel == 2 || FSel == 3 || FSel == 5 || FSel == 6 || FSel == 9 || FSel == 11 || FSel == 18)
			FDT = 3600*24; //1 day to develop the fault
		else if(FSel == 4 || FSel == 17)
			FDT = 3600; //1 Hour to develop the fault
		else if(FSel = 13)
			FDT = 60*15; // 15 minutes to develop the fault
		else
		{
			FSel = 20; //No fault selected
			FDT = 0;
			printf('Error! - Fault %d with type %d not specified for benchmark purpose')
			return;
		}
	}
	else
	{
		MFS = 0.25*FType;
		FDT = 0;

		if(FSel == 3 || FSel == 4 || FSel == 5 || FSel == 6 || FSel == 9)
		{
			FSel = 20; //No fault selected
			printf('Error! - Fault %d with type %d not specified for benchmark purpose')
			return;
		}
		else if(FSel == 2 || FSel == 11 || FSel == 15 || FSel == 17)
		{
			if(FType < 3)
			{
				FSel = 20; //No fault selected
				printf('Error! - Fault %d with type %d not specified for benchmark purpose')
				return;
			}
		}
	}

	if(FSel == 12 || FSel == 14 || FSel == 19)
		sParams.FSD = -1;

	if(FDT == 0)
		sParams.FDS = 999999999999999; //Very long number to simulate an inf slope
	else
		sParams.FDS = MFS/FDT;

	return sParams;
}

//Generate Gaussian distributed numbers with mean 0 and std 1 according to Knuth
/*To adjust to some other distribution, multiply by the standard deviation and add the mean.*/
double gaussrand()
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
int faultMap(int FSel)
{

	int nFaultTypes = 0;

	switch(FSel)
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







