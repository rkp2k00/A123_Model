# A123 DFN Model - 26650 M1B LFP Cell

Doyle–Fuller–Newman (DFN) electrochemical model parameterisation and optimisation for the A123 26650 M1B lithium iron phosphate cell (2.5 Ah), built with [PyBaMM](https://www.pybamm.org/).

---

## Repository Structure

```
A123_Model/
├── Document_report_A123.docx              # Final report documenting the full modelling workflow
├── Pybamm_parameters_initial.xlsx         # Initial DFN parameter compilation with literature references and calculation
├── references_papers/                     # Source papers used for parameterisation
├── pybamm_venv/                           # Virtual environment configuration for PyBaMM
└── notebooks/
    ├── data/
    ├── ocp_estimation/
    ├── python_files/
    ├── optimization_fit.png
    └── optimised_params.json
└── noteboooks_version_2/
    └── A123_DFN_2 (best HPPC result so far)/
    └── A123_DFN_3 (best crate results with optimizing the OCP's)
    └── Version_2_document (contains all the necessary information about the changes and results)
    └── Sensitivity analysis.

```

## Data

| File | Description |
|------|-------------|
| `A123_data_interpolated.xlsx` | Experimental voltage vs. time curves (1C, 1D, 2C, 3C, 4C) interpolated at 1 s intervals |
| `HPPC_data.xlsx` | Hybrid Pulse Power Characterisation (HPPC) test data |
| `OCP_Interpolated_Prada2012.xlsx` | Prada 2012 extracted LFP and Gr - Lithiation and Delithiation data |

## OCP Estimation

| File | Description |
|------|-------------|
| `Hunan_interpolated.xlsx` | Graphite OCP reference data used for MSMR fitting |
| `MSMR OCP estimation.ipynb` | Notebook covering the MSMR parameter fit for both LFP and graphite OCP |
| `msmr_opt.py` | Script for optimising MSMR gallery variables for a given electrode |

## Model version 1

The Initial DFN model was developed incrementally across four versions:

| Version | File | Description |
|---------|------|-------------|
| v1 | `DFN_a123_01.ipynb` | Baseline model with literature parameters |
| v2 | `DFN_a123_02.ipynb` | Manual adjustment of thermodynamic and transport parameters |
| v3 | `DFN_a123_OCV_optimized.ipynb` | MSMR-based graphite OCP and refined LFP OCP for improved full-cell OCV fit |
| v4 | `DFN_a123_final_opt.ipynb` | Automated optimisation of electrolyte, diffusivity, and kinetic parameters via `dfn_optimizer.py` |

## Model version 2

| Version | File | Description |
|---------|------|-------------|
| v5 | `A123_DFN_2.ipynb` | Best HPPC results after considering prada2012 OCP for lithiation and delithiation OCP's, adjusting electrode conductivity, bruggeman coefficient and other material as well as kinetic parameters |
| v6 | `A123_DFN_3.ipynb` | Best Crate results by adjusting OCP with current sigmoid hystersis model on both electrodes |

## Optimisation Outputs

| File | Description |
|------|-------------|
| `optimization_fit.png` | Comparison plots of simulated vs. experimental voltage for 1D, 1C, 2C, 3C, and 4C |
| `optimised_params.json` | Final optimised parameter values and scaling factors |

## Getting Started

```bash
# Activate the virtual environment
cd pybamm_venv
# Windows
.\Scripts\activate
# Linux/macOS
source bin/activate

# Run the latest model
cd notebooks/python_files
python model.py
```
