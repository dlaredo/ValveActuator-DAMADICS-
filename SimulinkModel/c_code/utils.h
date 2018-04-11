#ifndef _UTILS_H_
#define _UTILS_H_

int sendMail(char *);
void logMsg(FILE *, char *);
void getCurrentTime(char *);
int getProcessPid(char *);
void profileProcessMemory(int, char *);
int dbParamsFromString(char *, char *, char *, char *, char *);

#endif