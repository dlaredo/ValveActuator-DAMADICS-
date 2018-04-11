global dbconn

dbconn = database('damadics','dlaredorazo','@Dexsys13','com.mysql.jdbc.Driver','jdbc:mysql://localhost:3306/damadics'); 
today = datetime();


%colNames = {'timestamp', 'externalControllerOutput', 'undisturbedMediumFlow', 'pressureValveInlet', 'pressureValveOutlet', 'mediumTemperature', 'rodDisplacement', ...
%        'disturbedMediumFlow'};
    
%insertdata = [datestr(today), 0, 0, 0, 0, 0, 0, 1];

%fastinsert(dbconn,'valveReadings', colNames, insertdata);

%query = sprintf("INSERT INTO 'valveReadings' VALUES (%s, %f, %f, %f," + ...  
%    " %f, %f, %f, %f);", datestr(today), 0, 0, 0, 0, 0, 0, 1)
%exec(dbconn, query);