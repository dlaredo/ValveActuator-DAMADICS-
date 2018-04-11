function y = saveToDB(values)

time = values(10);

intTime=floor(time);
fracTime=time-intTime;

alreadySampled = evalin('base', 'alreadySampled');
sampleTime = 60;

%Write samples to the database
if fracTime == 0 && mod(intTime,sampleTime) == 0 && alreadySampled == 0
    alreadySampled = 1;
    assignin('base', 'alreadySampled', alreadySampled);
    conn = evalin('base', 'dbconn');
    writeToDB(conn, values(1:9), intTime);
elseif alreadySampled == 1 && mod(intTime,sampleTime) ~= 0
    alreadySampled = 0;
    assignin('base', 'alreadySampled', alreadySampled);
end

y = alreadySampled;

end

function writeToDB(dbConn, values, elapsedSeconds)

    controlValue = values(1);
    pressureValveInlet = values(2);
    pressureValveOutlet = values(3);
    rodDisplacement = values(4);
    disturbedMediumFlow = values(5);
    mediumTemperature = values(6);
    faultIntensity = values(7);
    selectedFault = values(8);
    faultType = values(9);
   
    startDateTime = evalin('base', 'lastSimulationDateTime');
    
    currentDateTime = updateDateTime(startDateTime, elapsedSeconds);
    
    logFile = evalin('base', 'logFileDescriptor');
    
    query = sprintf("INSERT INTO valveReadings(timestamp, externalControllerOutput, disturbedMediumFlow, "+...
    "pressureValveInlet, pressureValveOutlet, mediumTemperature, rodDisplacement, selectedFault, faultType, faultIntensity)"+ ...
    "VALUES ('%s', %f, %f, %f, %f, %f, %f, %f, %f, %f);", datestr(currentDateTime, 'yyyy-mm-dd HH:MM:SS'), ...
    controlValue, disturbedMediumFlow, pressureValveInlet, pressureValveOutlet, mediumTemperature, ...
    rodDisplacement, selectedFault, faultType, faultIntensity);
    curs = exec(dbConn, query);
    
    %If the insertion was successfull, keep track of the last inserted date
    %time
    if strcmp(curs.Message,'') == 1
        lastSimulationDateTime = currentDateTime;
        save('lastDateTime.mat', lastSimulationDateTime);
    else
        fprintf(logFile, "%s\n", curs.Message);
    end
    
end

function currentDateTime = updateDateTime(startDateTime, elapsedSeconds)

    runningHours = 0;
    runningMinutes = 0;
    runningSeconds = 0;

    runningDays = fix(elapsedSeconds/86400);
    reminder = mod(elapsedSeconds,86400);
    
    if reminder ~= 0
        
        runningHours = fix(reminder/3600);
        reminder = mod(reminder,3600);
        
        if reminder ~= 0
            runningMinutes = fix(reminder/60);
            runningSeconds = mod(reminder,60);
        end
        
    end
    
    currentDateTime = startDateTime;
    currentDateTime = currentDateTime + days(runningDays);
    currentDateTime = currentDateTime + hours(runningHours);
    currentDateTime = currentDateTime + minutes(runningMinutes);
    currentDateTime = currentDateTime + seconds(runningSeconds);
    
end
