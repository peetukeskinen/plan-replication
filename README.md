# Replication Files Finland's Net Expenditure Paths

## Overview of the European Commission DSA tool
The Debt Sustainability Analysis (DSA) tool is used for debt ratio projections. This version incorporates all criteria from the reformed EU Fiscal Rules. It improves the analysis of debt sustainability, including new fiscal safeguards, as outlined by the European Commission. For context on the importance of these debt rules for Finland, see this [report, ch. 3]([https://www.vtv.fi/en/blog/the-length-of-the-adjustment-plan-in-the-reformed-eu-debt-rules-is-of-great-importance-to-finland/](https://www.vtv.fi/en/publications/fiscal-policy-monitoring-report-2024/)). Current version also incorporates the calculation of net expenditure path.

The code has benefitted greatly from the analysis and Python code by Darvas et al. (2023), as seen [here](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) and [here](https://github.com/lennardwelslau/eu-debt-sustainability-analysis).

## Replication of Results
Below are instructions on which MATLAB “settings” files to **run** in order to reproduce the net expenditure paths as reported by the European Commission.
Commission results using spring 2024 data for DSA can be seen from this [Excel file](https://economy-finance.ec.europa.eu/document/download/d3b7feec-2544-40d2-8b94-198479c66755_en?filename=Commission_prior_guidance_calculation_sheet_finland.xlsx).

1. **Commission Reference Trajectory**  
   - **Settings file:**  `ReferenceTrajectorySettings.m`  
   - **Expected output:** Net expenditure path using spring 2024 data and linear adjustment identical to the Commission’s “reference trajectory” results, as reported in Table 1 of the following file.
   - **Commission source:**  
     [Table 1: Reference trajectory provided by the Commission to Finland on 21 June 2024](https://economy-finance.ec.europa.eu/document/download/2685c07d-ce5e-41aa-abb8-6ef14d0e72d9_en?filename=MTFSP_2025_FI.pdf)

2. **Finland’s Net Expenditure Path in its Medium-Term Plan (Safeguards binding)**  
   - **Settings file:**  `FinlandPlansettings.m`  
   - **Expected output:** Net expenditure path using Finland's medium term plan data (autumn 2024) and frontloaded adjustment matching exactly those in Table 2a (1st row) of the following file.
   - **Commission source:**  
     [Table 2: Maximum net expenditure growth of Finland (a)](https://economy-finance.ec.europa.eu/document/download/2685c07d-ce5e-41aa-abb8-6ef14d0e72d9_en?filename=MTFSP_2025_FI.pdf)

3. **Finland’s Net Expenditure Path with National Escape Clause (Safeguards not binding)**  
   - **Settings file:** `NoSafeguardsSettings.m`  
   - **Expected output:** Net expenditure path using Finland's medium term plan data (autumn 2024) and linear adjustment matching exactly those in Table 2b (1st row) of the Commission’s “Non-Compliance” documentation for Finland. In this replication, we assume that Commission uses linear adjustment. Commission has not reported the methodology used in their calculations in this scenario.
   - **Commission source:**  
     [Table 2: Maximum net expenditure growth, Net of the impact of the debt sustainability safeguard and the deficit resilience safeguard (b)](https://economy-finance.ec.europa.eu/document/download/a15e0f75-3100-42c5-bb7e-f0ea5819ffa6_en?filename=FI_NEC_COM_2025_606_1_EN_ACT_part1_v3.pdf)

### Components Required
To execute this MATLAB code, you'll need:

0. **The Run File:**  `ReferenceTrajectorySettings.m` and `FinlandPlansettings.m` and `NoSafeguardsSettings.m`  
1. **Main Function:** `runDsaModel5_1_2.m`  
2. **Helper Functions:**  
   - `project_debt5_1_2v.m` – Projects debt paths considering yearly adjustments.  
   - `sumq2y.m` – Converts quarterly shocks to yearly data.  
   - `formatWithSpaces.m` – Ensures numbers in figures are formatted for readability.  
3. **Data Files:** `CommissionData.xlsx` and  `FinlandMediumTermPlanData.xlsx`

Save above mentioned files in the same folder and run one of the Run Files in part 0 to execute the MATLAB code.

### Criteria
This version includes all criteria from the reformed EU fiscal rules, including:

- **DSA-based Criteria:** Both deterministic and stochastic scenarios.  
- **Debt Sustainability Safeguard**  
- **Deficit Resilience Safeguard**  
- **Deficit Benchmark**

### Compatibility
The tool is compatible with Windows 10 (64-bit) and MATLAB R2020b.

### Contact
For any inquiries or feedback, please contact peetu.keskinen@vtv.fi.
