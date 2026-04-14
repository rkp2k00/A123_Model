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

## OCP Estimation

| File | Description |
|------|-------------|
| `Hunan_interpolated.xlsx` | Graphite OCP reference data used for MSMR fitting |
| `MSMR OCP estimation.ipynb` | Notebook covering the MSMR parameter fit for both LFP and graphite OCP |
| `msmr_opt.py` | Script for optimising MSMR gallery variables for a given electrode |

## Model Evolution

The DFN model was developed incrementally across four versions:

| Version | File | Description |
|---------|------|-------------|
| v1 | `DFN_a123_01.py` | Baseline model with literature parameters |
| v2 | `DFN_a123_02.py` | Manual adjustment of thermodynamic and transport parameters |
| v3 | `DFN_a123_OCV_optimized.py` | MSMR-based graphite OCP and refined LFP OCP for improved full-cell OCV fit |
| v4 | `DFN_a123_final_opt.py` | Automated optimisation of electrolyte, diffusivity, and kinetic parameters via `dfn_optimizer.py` |

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
