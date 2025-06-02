%% PARAMETER SELECTION OPTIONS

   % SELECT ADJUSTMENT PLAN LENGTH:
        %  4) 4-year plan
        %  7) 7-year plan

    % SELECT SFA METHOD:
        %  0) COM New Revised Assumption
        % -1) COM Old Zero Assumption

    % IMPOSE DEBT SUSTAINABILITY SAFEGUARD:
        % 1) yes, 0) no 

    % IMPOSE DEFICIT RESILIENCE SAFEGUARD:
        % 1) yes, 0) no 

    % PLOTTING:
        % 1) yes, 0) no 

    % STOCHASTIC SAMPLES (Power of ten for simulated paths):
        % 3) 1,000 simulated paths
        % 4) 10,000 simulated paths
        % 5) 100,000 simulated paths
        % 6) 1,000,000 simulated paths

    % PLAUSIBILITY VALUE:
        % 7) 70%
        % 8) 80%
        % 9) 90%

    % LANGUAGE FOR STOCHASTIC PLOTS:
        % 1) English
        % 2) Suomi (Finnish)

    % STOCHASTIC METHOD:
        % 1) Normal Distribution Simulation
        % 2) Bootstrap Simulation shock as deviations

    % SAVE RESULTS:
        % 1) Save as .mat file
        % 0) Do not save the results as .mat file
       
    % WEIGHT OF THE ADJUSTMENT

%% Define parameters in a structure for runDsaModel5_1.m
params = struct;
params.adjustmentPeriods = 7;
params.sfa_method = 0;
params.apply_debt_safeguard = 0;
params.apply_deficit_benchmark =1;
params.apply_deficit_safeguard = 1;
params.plotting = 0;
params.power = 4;
params.plausibility = 7;
params.language = 1;
params.stoch_method = 1;
params.saveFlag = 0;
%params.w_adjustment = zeros(params.adjustmentPeriods,1);

% Finland's front-loaded adjustment (change in 
% structural primary balance)
params.w_adjustment = [0.81-.27; 0.65-.27; 0.31-.27;...
                        0.35-.27; 0.27-.27; 0.27-.27; 0.27-.27];
      
%% Choose country data file

% Data from Finland's medium term plan
% https://julkaisut.valtioneuvosto.fi/handle/10024/165863
params.fileName = 'CommissionPriorGuidanceFinland_suunnitelma.xlsx'; % Finland

%% Call the function with the parameter structure (with initial debt taken from excel)        
runDsaModel5_1_2(params);
  