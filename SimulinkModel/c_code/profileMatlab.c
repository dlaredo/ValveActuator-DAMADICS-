#include <stdio.h>
#include <unistd.h>
#include "utils.h"

#define PROFILE_TIME 900 //In seconds

int main()
{
	int matlabId = 0;
	char memoryStats[64] = {};
	FILE *fp = NULL;

	if((fp = fopen("matlabMemoryStats.txt", "w")) == NULL)
	{
		printf("Can not create logFile\n");
		return -1;
	}

	fprintf(fp, "%15s %17s %15s %10s %12s %15s\n\n", "Time", "Size", "Resident", "Shared", "Text", "Data/Stack");

	//Profile matlab memory while it is open
	while((matlabId = getProcessPid("MATLAB")) != 0)
	{
		profileProcessMemory(matlabId, memoryStats);
		logMsg(fp, memoryStats);
		sleep(PROFILE_TIME);
	}

	fclose(fp);

	return 0;
}