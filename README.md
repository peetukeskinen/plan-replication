# Replication files Finland's net expenditure paths

## Overview
The Debt Sustainability Analysis (DSA) tool is used for debt ratio projections. This version incorporates all criteria from the reformed EU Fiscal Rules. It improves the analysis of debt sustainability, including new fiscal safeguards, as outlined by the European Commission. For context on the importance of these debt rules, see this [blog post](https://www.vtv.fi/en/blog/the-length-of-the-adjustment-plan-in-the-reformed-eu-debt-rules-is-of-great-importance-to-finland/). Current version also incorporates the calculation of net expenditure path.

The code has benefitted greatly from the analysis and Python code by Darvas et al. (2023), as seen [here](https://www.bruegel.org/working-paper/quantitative-evaluation-european-commissions-fiscal-governance-proposal) and [here](https://github.com/lennardwelslau/eu-debt-sustainability-analysis).

## Replication of Results
Below are instructions on which MATLAB “settings” files to **run** in order to reproduce the net expenditure paths (and medium-term plan figures) as reported by the European Commission.

1. **Commission Reference Trajectory**  
   - **Settings file:**  `ReferenceTrajectorySettings.m`  
   - **Expected output:** Net expenditure path identical to the Commission’s “reference trajectory” results, as reported in Table 1 of the Commission’s 2025 Medium-Term Fiscal-Structural Plan for Finland.  
   - **Commission source:**  
     [MTFSP 2025 FI – Table 1](https://economy-finance.ec.europa.eu/document/download/2685c07d-ce5e-41aa-abb8-6ef14d0e72d9_en?filename=MTFSP_2025_FI.pdf)

2. **Finland’s Medium-Term Plan (Safeguards binding)**  
   - **Settings file:**  `FinlandPlansettings.m`  
   - **Expected output:** Medium-term plan figures matching exactly those in Table 2 (first row) of the Commission’s 2025 Medium-Term Fiscal-Structural Plan for Finland.  
   - **Commission source:**  
     [MTFSP 2025 FI – Table 2 (first row)](https://economy-finance.ec.europa.eu/document/download/2685c07d-ce5e-41aa-abb8-6ef14d0e72d9_en?filename=MTFSP_2025_FI.pdf)

3. **Finland’s Medium-Term Plan (Safeguards not binding)**  
   - **Settings file:** `NoSafeguardsSettings.m`  
   - **Expected output:** Medium-term plan figures matching exactly those in Table 2 (third row) of the Commission’s “Non-Compliance” documentation for Finland.  
   - **Commission source:**  
     [FI Non-Compliance 2025 – Table 2 (third row)](https://economy-finance.ec.europa.eu/document/download/a15e0f75-3100-42c5-bb7e-f0ea5819ffa6_en?filename=FI_NEC_COM_2025_606_1_EN_ACT_part1_v3.pdf)

### Components Required
To execute this MATLAB code, you'll need:

0. **The Run File:**  `ReferenceTrajectorySettings.m` and `FinlandPlansettings.m` and `NoSafeguardsSettings.m`  
1. **Main Function:** `runDsaModel5_1_2.m`  
2. **Helper Functions:**  
   - `project_debt5_1_2v.m` – Projects debt paths considering yearly adjustments.  
   - `sumq2y.m` – Converts quarterly shocks to yearly data.  
   - `formatWithSpaces.m` – Ensures numbers in figures are formatted for readability.  
3. **Data Files:** `CommissionData.xlsx` and  `FinlandMediumTermPlanData.xlsx`

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
