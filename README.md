# Debt Sustainability Analysis (DSA) Tool - Version 5.1.2

## Overview

The Debt Sustainability Analysis (DSA) tool is used for projecting debt-to-GDP ratios under different fiscal scenarios. This version (5.1.2) includes all criteria from the reformed EU fiscal rules and provides improved modeling of debt sustainability, including safeguards introduced by the European Commission. It also supports net expenditure path calculations.

For background on the new rules and their significance for Finland, see this [blog post](https://www.vtv.fi/en/blog/the-length-of-the-adjustment-plan-in-the-reformed-eu-debt-rules-is-of-great-importance-to-finland/).

The code builds on the approach and Python scripts developed by Darvas et al. (2023), available [here](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) and [GitHub repo](https://github.com/lennardwelslau/eu-debt-sustainability-analysis).

## Compatibility

- MATLAB R2020b (64-bit)
- Windows 10 (64-bit)

## Files Included

To use this tool, the following files are required:

- **Main Configuration File**
  - `defineDsaModel5_1_2.m` – defines the `param` structure with user inputs and settings

- **Core Analysis File**
  - `runDsaModel5_1_2.m` – runs the DSA using the provided parameters

- **Helper Functions**
  - `project_debt5_1_2v.m` – core logic for projecting the debt path
  - `sumq2y.m` – converts quarterly shocks to annual format
  - `formatWithSpaces.m` – formats numerical outputs for readability

- **Data File**
  - `CommissionPriorGuidanceFinland_suunnitelma.xlsx` – contains baseline projections and fiscal assumptions from Finland's medium term plan. 
    ([source](https://economy-finance.ec.europa.eu/economic-and-fiscal-governance/stability-and-growth-pact/preventive-arm/national-medium-term-fiscal-structural-plans_en))

## EU Fiscal Rule Criteria Included

Version 5.1.2 implements the following rules as defined in the reformed EU framework:

- **DSA-Based Criteria**
  - Deterministic debt sustainability analysis (baseline + adverse)
  - Stochastic (fan chart) simulations

- **Debt Sustainability Safeguard**
  - Ensures declining debt trajectory under standard assumptions

- **Deficit Resilience Safeguard**
  - Evaluates whether the structural deficit remains below the 3% Maastricht threshold

- **Deficit Benchmark**
  - Requires minimum fiscal effort consistent with compliance

## Customization Options

All user-defined settings are located in `defineDsaModel5_1.m`. These include:

- Projection horizon
- Interest-growth rate differential
- Target debt or deficit levels
- Stochastic shock settings
- Plotting and language preferences
- Adjustment weights (see below)

### Adjustment Path Weights (`params.w_adjustment`)

You can define a non-linear adjustment path using the `params.w_adjustment` field.

By default, this is set to zero, which results in a **linear adjustment path** for the Structural Primary Balance (SPB). If you want to replicate a **front-loaded adjustment** as used in Finland’s medium-term plans, uncomment and use the following:

```matlab
% params.w_adjustment = [0.54; 0.38; 0.04; ...
%                        0.08; 0.00; 0.00; 0.00];

### Data and Adjustments
The file `CommissionPriorGuidanceFinland.xlsx` contains all necessary data for the tool. Users can modify parameters for sensitivity analysis and select options for plotting, language preference, and saving.

### Example

To execute the tool with selected configurations, modify the parameters in the `defineDsaModel5_1.m` file as needed and run the file. The selected parameter structure is passed to the main function `runDsaModel5_1.m` for analysis.

### Contact
For any inquiries or feedback, please contact peetu.keskinen@vtv.fi.
