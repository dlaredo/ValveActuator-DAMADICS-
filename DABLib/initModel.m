function initModel()

    FSel = 20;
    ftype = 1;
    faultInProcess = 0;
    
    alreadySampled = 0;
    
    keySet = linspace(1,19,19);
    valueSet = {[1,2,3], [3,4], [4], [4], [4], [4], [1,2,3], [1,2,3],...
        [4], [1,2,3], [3,4], [1,2,3], [1,2,3,4], [1,2,3], [3], [1,2,3], ...
        [3,4], [1,2,3,4], [1,2,3]};

    fmap = containers.Map(keySet,valueSet);
    
    startTime = 0;
    
    %Variables for the fault behavior. Initialized as DGen block Init
    DSim=0;
    FSD=1;
    FFrom = 9;
    FTo = inf;
    FMFS=1;
    FFDT = 0;
    FDS = 99999999999999999999999999;
    From = FFrom;
    
    %Database initialization
    dbconn = database('damadics','dlaredorazo','@Dexsys13','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/damadics'); 
    simulationDateTime = datetime();
    timeLimit = 2592000; %equivalent to 30 days
    %timeLimit = 20;
    
    logFile = fopen('faultGeneratorLog.txt', 'a+');
    
    %create model variables in the workspace
    assignin('base', 'DSim', DSim);
    assignin('base', 'faultInProcess', faultInProcess);
    assignin('base', 'FDS', FDS);
    assignin('base', 'FFDT', FFDT);
    assignin('base', 'FFrom', FFrom);
    assignin('base', 'fmap', fmap);
    assignin('base', 'FMFS', FMFS);
    assignin('base', 'From', From);
    assignin('base', 'FSD', FSD);
    assignin('base', 'FSel', FSel);
    assignin('base', 'FTo', FTo);
    assignin('base', 'ftype', ftype);
    assignin('base', 'startTime', startTime);
    assignin('base', 'alreadySampled', alreadySampled);
    assignin('base', 'dbconn', dbconn);
    assignin('base', 'lastSimulationDateTime', simulationDateTime);
    assignin('base', 'timeLimit', timeLimit);
    assignin('base', 'logFileDescriptor', logFile);
    
    assignin('base', 'counter', 0);

end

%[FSel, ftype, faultInProcess, fmap, startTime, DSim, FFDT, FFrom, FMFS, FSD, FTo, FDS, From, alreadySampled, dbconn, today] = initModel()