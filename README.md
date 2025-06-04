# Debt Sustainability Analysis (DSA) Tool - Version 5.1.2

## Overview
The Debt Sustainability Analysis (DSA) tool is used for debt ratio projections. This version incorporates all criteria from the reformed EU Fiscal Rules. It improves the analysis of debt sustainability, including new fiscal safeguards, as outlined by the European Commission. For context on the importance of these debt rules, see this [blog post](https://www.vtv.fi/en/blog/the-length-of-the-adjustment-plan-in-the-reformed-eu-debt-rules-is-of-great-importance-to-finland/). **Current version also incorporates the calculation of net expenditure path.**

The code has benefitted greatly from the analysis and Python code by Darvas et al. (2023), as seen [here](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) and [here](https://github.com/lennardwelslau/eu-debt-sustainability-analysis).

### Compatibility
The tool is compatible with Windows 10 (64-bit) and MATLAB R2020b.

### Components Required
To execute this MATLAB code, you'll need:

0. **The Run File:**  `defineDsaModel5_1.m`  
1. **Main Function:** `runDsaModel5_1.m`  
2. **Helper Functions:**
   - `project_debt5_1v.m` – Projects debt paths considering yearly adjustments.
   - `sumq2y.m` – Converts quarterly shocks to yearly data.
   - `formatWithSpaces.m` – Ensures numbers in figures are formatted for readability.  
3. **Data File:** `CommissionPriorGuidanceFinland.xlsx`  
   ([source](https://economy-finance.ec.europa.eu/economic-and-fiscal-governance/stability-and-growth-pact/preventive-arm/national-medium-term-fiscal-structural-plans_en))

### Criteria
The current version 5.1 includes all criteria from the reformed EU fiscal rules, including:

- **DSA-based Criteria:** Both deterministic and stochastic scenarios.
- **Debt Sustainability Safeguard**
- **Deficit Resilience Safeguard**
- **Deficit Benchmark**

These criteria ensure that the analysis aligns with updated EU regulations and is more comprehensive than previous versions.

### Scenarios and Customization
The tool facilitates debt projections following the guidelines of the European Commission's [Debt Sustainability Monitor 2023](https://economy-finance.ec.europa.eu/publications/debt-sustainability-monitor-2023_en). Users can run simulations under different assumptions and fiscal conditions, with flexibility in selecting methods and parameters for more customized results.

The use of the tool is done by running a separate file, `defineDsaModel5_1.m`, where all required parameters and options are selected. The defined `param` structure is then passed to the `runDsaModel5_1.m` function to run the analysis.

#### Adjustment Path Weights (`params.w_adjustment`)
You can define a non-linear adjustment path using the `params.w_adjustment` field. By default, this is set to zero, which results in a linear adjustment path for the Structural Primary Balance (SPB). If you want to replicate a front-loaded adjustment as used in Finland’s medium-term plans, uncomment and use the following:

```matlab
% params.w_adjustment = [0.81-.]()_
