function [FSel, ftype] = RandomFaultGenerator(currentTime, DGenBlockAddress)

    %Variables used for the random generation of the fault
    muGen = 0;
    stdGen = 1;
    threshGen = 5.7;
    
    %Variables used for the stopping of the fault
    muStop = 0;
    stdStop = 1;
    threshStop = 2;
    
    %counter = evalin('base', 'counter');
    %counter = counter+1;
    %assignin('base', 'counter', counter);
    
    %offsetTime = 100; %In seconds to compensate for the overhead
    offsetTime = 1; %In seconds to compensate for the overhead
    logFile = evalin('base', 'logFileDescriptor');
    
    %Based on Damadics Benchmark Definition V1_0
    horSeconds = [1500, 2400, 5100, 85500]; 
    %horSeconds = [15, 24, 51, 85];
    
    FSel = evalin('base', 'FSel');
    ftype = evalin('base', 'ftype');
    faultInProcess = evalin('base', 'faultInProcess');
    timeLimit = evalin('base', 'timeLimit');
    
    %If time limit reached stop simulation
    if  currentTime > timeLimit
        
        %set simulation running flag to 0 so that the script can re run the
        %simulation
        simulationRunning = 0;
        assignin('base', 'simulationRunning', 0);
        dbConn = evalin('base', 'dbconn');
        
        DSim = 1;
        fprintf(logFile, 'Stopping simulation after %f seconds of running\n', currentTime);
        
        %close log file and database connection
        fclose(logFile);
        close(dbConn);
        %stop simulation
        set_param(strcat(DGenBlockAddress, '/Disable simulation'), 'value', num2str(DSim));
        warning on;
        return;
    end
    
    %Only one fault at the time
    if faultInProcess == 0
        b = generateRandomFault(muGen, stdGen, threshGen);
        
        %If no fault in process and generateRandomFault indicates that a
        %fault should be generated
        %b = 0; %Dont generate faults yet.
        if b == 1
            
            %set_param('DRandomGen/FaultInProcess', 'Value', 1);
            
            faultInProcess = 1;
            [FSel, ftype] = getRandomFault();
            [DSim, FSD, FFrom, FTo, FMFS, FFDT] = getSimulationParameters(FSel, ftype, currentTime, offsetTime);
            [FDS, To, FSD] = getSignalParameters(FFDT, FMFS, FTo, FSD);

            setSimulationParameters(DGenBlockAddress, FSel, FFrom, FDS, To, FSD);
            
            fprintf(logFile, 'Generating Fault: %d of Type:%d at time %f\n', FSel, ftype, currentTime);
            fprintf(logFile, 'Simulation parameters: From: %d, FDS:%d, To:%f, FSD:%f\n', FFrom, FDS, To, FSD);
            fprintf(2, 'Generating Fault: %d of Type:%d at time %f', FSel, ftype, currentTime);
            assignin('base', 'startTime', currentTime);
        end
    else
        %If there is a fault in process, see if its time to stop it.
        
        startTime = evalin('base', 'startTime');
        elapsedSeconds = currentTime - startTime;

        %Only stop if the fault has completely developed according to the
        %fault type
        horizonSeconds = horSeconds(ftype);
        if elapsedSeconds > horizonSeconds
            
             b = stopFault(muStop, stdStop, threshStop);
             if b == 1
                 
                 fprintf(logFile, 'Stopping Fault: %d of Type:%d at time %f\n', FSel, ftype, currentTime);
                 fprintf(2, 'Stopping Fault: %d of Type:%d at time %f\n', FSel, ftype, elapsedSeconds);
                 faultInProcess = 0;
                 FSel = 20;
                 ftype = 1;
                 
             end
            
        end
        
    end

    assignin('base', 'faultInProcess', faultInProcess);
    assignin('base', 'FSel', FSel);
    assignin('base', 'ftype', ftype);

end

function b = generateRandomFault(mu, std, thresh)
%Draw a random number from a normal distribution with mean mu and 
%standard deviation std. If the number is higher than threshold thres,
%then generate a random fault.

    b = 0;

    r = normrnd(mu,std);
    if r > thresh
        b = 1;
    end
    
end

function b = stopFault(mu, std, thresh)
%Draw a random number from a normal distribution with mean mu and 
%standard deviation std. If the number is higher than threshold thres,
%then stop random fault.

    b = 0;

    r = normrnd(mu,std);
    if r > thresh
        b = 1;
    end
    
end

function [fsel, ftype] = getRandomFault()

    faultMap = evalin('base', 'fmap');
    fsel = randi(19,1);
    
    ftypes = faultMap(fsel);
    asize = max(size(ftypes));
    
    rftype = randi(asize,1);
    ftype = ftypes(rftype);

end

%Compute the elapsed seconds between the time the fault started and the
%current time
function elapsedSeconds = computeElapsedSeconds(startDateTime, currentDateTime)

    elapsedYears = currentDateTime.Year - startDateTime.Year;
    elapsedMonths = currentDateTime.Month - startDateTime.Month;
    elapsedDays = currentDateTime.Day - startDateTime.Day;
    elapsedHours = currentDateTime.Hour - startDateTime.Hour;
    elapsedMinutes = currentDateTime.Minute - startDateTime.Minute;
    elapsedSeconds = currentDateTime.Second - startDateTime.Second;
    
    elapsedSeconds = elapsedDays*86400+elapsedHours*3600+...
        elapsedMinutes*60+elapsedSeconds;
    
end

%Set the parameters for the simulation of a specified fault. Taken from
%DGen mask init
function [DSim, FSD, FFrom, FTo, FMFS, FFDT] = getSimulationParameters(FSel, FType, currentTime, offsetTime)

    DSim=0;
    FSD=1;
    FFrom = evalin('base', 'From');
    FTo = inf;
    if FType==4
        FMFS=1;
        switch FSel
            case {2, 3, 5, 6, 9, 11, 18}
                FFDT = 3600*24;
                %FFDT = 3.6*24;
            case {4, 17}
                FFDT = 3600;
                %FFDT = 3.6;
            case {13}
                FFDT = 60*15;
                %FFDT = 6*1.5;
            case {20}
                FFDT = 0;
            otherwise
                FSel = 20;
                FFDT = 0;
                DSim=1;
                 fprintf(2, 'Error! - Fault with selected type not specified for benchmark purpose.')
        end
    else
        FMFS=0.25*FType;
        FFDT=0;
        switch FSel
            case {3, 4, 5, 6, 9}
                DSim=1;
                fprintf(2, 'Error! - Fault with selected type not specified for benchmark purpose.')
            case {2, 11, 15, 17}
                if FType<3
                    DSim=1;
                     fprintf(2, 'Error! - Fault with selected type not specified for benchmark purpose.')
                end
            otherwise
        end
    end
    switch FSel
        case {12, 14, 19}
            FSD = -1;
        otherwise
    end
    
    %Recompute FFrom according to the current time plus an offset to
    %compensate for the overhead
    FFrom = currentTime + offsetTime;
    
end

%Set the parameters that will generate the fault for a specified set of
%fault parameters. Taken from FGen mask
function [FDS, To, FSD] =  getSignalParameters(FDT, MFS, To, FSD)

    if FDT==0
        FDS = 99999999999999999999999999;
    else
        FDS = MFS / FDT;
    end
    
    if To==inf
        To = 99999999999999999999999999;
    end
    
    if FSD >= 0
        FSD=1;
    else
        FSD=-1;
    end

end

%Change the simulation parameters in the model so that the new scenario
%takes place.
function setSimulationParameters(DGenBlockAddress, FSel, FFrom, FDS, To, FSD)

    set_param(strcat(DGenBlockAddress, '/FGen/Ramp'), 'start', num2str(FFrom));
    set_param(strcat(DGenBlockAddress, '/FGen/Ramp'), 'slope', num2str(FDS));
    set_param(strcat(DGenBlockAddress, '/FGen/Step'), 'time', num2str(To));
    set_param(strcat(DGenBlockAddress, '/FGen/Constant5'), 'value', num2str(FSD));
    set_param(strcat(DGenBlockAddress, '/Fault Selector/Constant4'), 'value', num2str(FSel));

end