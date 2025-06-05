%% PARAMETER SELECTION OPTIONS
params = struct;

%% Key parameters

   % SELECT ADJUSTMENT PLAN LENGTH:
        %  4) 4-year plan
        %  7) 7-year plan
params.adjustmentPeriods = 7;

    % ADJUSTMENT PROFILE
        % 1) Linear adjustment + Commission data
        % 2) Frontloaded adjustment + Finland's plan data
        % 3) Linear adjustment + Finland's plan data
params.adjustment_profile = 2;

    % IMPOSE DEBT SUSTAINABILITY SAFEGUARD:
        % 1) yes, 0) no 
params.apply_debt_safeguard = 1;

    % IMPOSE DEBT SUSTAINABILITY SAFEGUARD:
        % 1) yes, 0) no 
params.apply_deficit_benchmark =0;        
             
    % IMPOSE DEFICIT RESILIENCE SAFEGUARD:
        % 1) yes, 0) no 
params.apply_deficit_safeguard = 1;



%% Other parameters

    % PLOTTING:
        % 1) yes, 0) no 
params.plotting = 0;

    % STOCHASTIC SAMPLES (Power of ten for simulated paths):
        % 3) 1,000 simulated paths
        % 4) 10,000 simulated paths
        % 5) 100,000 simulated paths
        % 6) 1,000,000 simulated paths
params.power = 3;

    % PLAUSIBILITY VALUE:
        % 7) 70%
        % 8) 80%
        % 9) 90%
params.plausibility = 7;

    % LANGUAGE FOR STOCHASTIC PLOTS:
        % 1) English
        % 2) Suomi (Finnish)
params.language = 1;

    % STOCHASTIC METHOD:
        % 1) Normal Distribution Simulation
        % 2) Bootstrap Simulation shock as deviations
params.stoch_method = 1;

    % SAVE RESULTS:
        % 1) Save as .mat file
        % 0) Do not save the results as .mat file
params.saveFlag = 0;     
        
    % SELECT SFA METHOD:
        %  0) COM New Revised Assumption
        % -1) COM Old Zero Assumption
params.sfa_method = 0;


%% Call the function with the parameter structure (with initial debt taken from excel)        
runDsaModel5_1_2(params);
  