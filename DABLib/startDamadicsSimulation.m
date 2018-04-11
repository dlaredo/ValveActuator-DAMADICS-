function startDamadicsSimulation()

    simulationRunning = 0;
    assignin('base', 'simulationRunning',simulationRunning);
    
    while true
        
        simulationRunning = evalin('base', 'simulationRunning');
        fprintf(2, "simRunning: %d", simulationRunning);
        
        if simulationRunning == 0

            DGenBlockAddress = 'RDataGeneration/RDGen';

            initModel();

            if exist('lastDateTime.mat', 'file') == 2
                load('lastDateTime.mat', 'lastSimulationDateTime')
            else
                lastSimulationDateTime = datetime();
                save('lastDateTime.mat', 'lastSimulationDateTime');
            end

            assignin('base', 'lastSimulationDateTime', lastSimulationDateTime+seconds(5));

            set_param(strcat(DGenBlockAddress, '/Disable simulation'), 'value', '0');
            fprintf(2, 'starting simulation');
            sim('RDataGeneration');
            simulationRunning = 1;
            assignin('base', 'simulationRunning', simulationRunning);

            clc;
        else
            pause(10); %sleep for 10 minutes
            fprintf(2, 'awaking');
        end
    
    end
    
end