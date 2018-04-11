function startDamadicsSimulation2()


            DGenBlockAddress = 'RDataGeneration/RDGen';

            initModel();
            
            warning off;
            
            sampleTime = 60;

            if exist('lastDateTime.mat', 'file') == 2
                load('lastDateTime.mat', 'lastSimulationDateTime')
            else
                lastSimulationDateTime = datetime();
                save('lastDateTime.mat', 'lastSimulationDateTime');
            end

            lastSimulationDateTime = lastSimulationDateTime+seconds(sampleTime);
            assignin('base', 'lastSimulationDateTime', lastSimulationDateTime);

            set_param(strcat(DGenBlockAddress, '/Disable simulation'), 'value', '0');
            
            
            logFile = evalin('base', 'logFileDescriptor');
            fprintf(logFile, 'Starting simulation on  %s\n', datestr(lastSimulationDateTime));
            sim('RDataGeneration', 'StopTime', 'inf', 'SimulationMode', 'accelerator');

            clc;
    
end