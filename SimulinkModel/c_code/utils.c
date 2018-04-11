#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "utils.h"

int sendMail(char *msg){

	char cmd[100];  // to hold the command.
    char to[] = "dlaredorazo@ucmerced.edu"; // email id of the recepient.
    char from[] = "dlaredorazo@gmail.com";
    char subject[] = "Simulation failure";    // email body.
    char tempFile[100];     // name of tempfile.
    char template_name[]="/tmp/smailXXXXXX";
    int fd;

    if((fd = mkstemp(template_name)) == -1)  //Create tmp file
        return -1;

    //strcpy(tempFile,tempnam("/tmp","sendmail")); // generate temp file name.

    //FILE *fp = fopen(tempFile,"w"); // open it for writing.
    FILE *fp = fdopen(fd, "w"); // open it for writing.
    
    if(fp == NULL)
        return -1;

    //unlink(template_name);

    fprintf(fp, "To: %s\nFrom: %s\nSubject: %s\n\n", to, from, subject);

    fprintf(fp,"%s\n", msg); // write body to it.

    fclose(fp);             // close it.

    sprintf(cmd,"ssmtp %s < %s",to,template_name); // prepare command.
    system(cmd);     // execute it.

    unlink(template_name); //Unlink it so that it gets closed

    return 0;

}

void getCurrentTime(char *timeString){

	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	sprintf(timeString, "%02d/%02d/%04d %02d:%02d:%02d", timeinfo->tm_mon, timeinfo->tm_mday, 
		1900+timeinfo->tm_year, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
}

void logMsg(FILE *logfile, char *msg){

	char logMsg[512] = {};
	char timeString[128] = {};

	getCurrentTime(timeString);
	strcpy(logMsg, timeString);
	strcat(logMsg, "\t");
	strcat(logMsg, msg);

	fprintf(logfile, "%s\n", logMsg);
	fflush(logfile);

}

int getProcessPid(char *processName){

	char command[100] = {};
	FILE* fp = NULL;
    int pid = 0;

    sprintf(command, "pgrep -x %s", processName);

    if((fp=popen(command, "r")) != NULL)
    {
    	fscanf(fp, "%d", &pid);
    	pclose(fp);
    }

    return pid;
}

void profileProcessMemory(int pid, char *results){

	char command[100] = {};
	FILE* fp = NULL;
    float pageToMbFactor = 4.0/1024.0;

    float size = 0, resident = 0, shared = 0, lib = 0, text = 0, data_stack = 0, dt = 0;

    sprintf(command, "cat /proc/%d/statm", pid);

    if((fp=popen(command, "r")) != NULL)
    {
    	fscanf(fp, "%f %f %f %f %f %f %f", &size, &resident, &shared, &text, &lib, &data_stack, &dt);
    	pclose(fp);
    }

    sprintf(results, "%12.4f %12.4f %12.4f %12.4f %12.4f", size*pageToMbFactor, resident*pageToMbFactor, 
    	shared*pageToMbFactor, text*pageToMbFactor, data_stack*pageToMbFactor);

}

int dbParamsFromString(char *rawString, char *host, char *user, char *pass, char *db)
{

    int slen = strlen(rawString), i = 0, j = 0;
    char auxBuff[256] = {};
    char auxChar;
    int hostParsed = 0, userParsed = 0, passParsed = 0, dbParsed = 0;

    for(i = 0; i < slen; i++)
    {
        auxChar = rawString[i];

        if(userParsed == 0)
        {
            if(auxChar != '@')
            {
                user[j] = auxChar;
                j++;
            }
            else
            {
                user[j] = '\0';
                userParsed = 1;
                j = 0;
            }
        }
        else if(hostParsed == 0)
        {
            if(auxChar != ' ')
            {
                host[j] = auxChar;
                j++;
            }
            else
            {
                host[j] = '\0';
                hostParsed = 1;
                j = 0;
            }
        }
        else if(passParsed == 0)
        {
            if(auxChar != ' ')
            {
                pass[j] = auxChar;
                j++;
            }
            else
            {
                pass[j] = '\0';
                passParsed = 1;
                j = 0;
            }
        }
        else
        {
            if(auxChar != '\0')
            {
                db[j] = auxChar;
                j++;
            }
            else
            {
                db[j] = '\0';
                dbParsed = 1;
                j = 0;
            }
        }
    }

    if (hostParsed+userParsed+passParsed+dbParsed != 4)
        return -1;

    return 0;

}




