"""
DFN Model Parameter Optimizer for LFP High-Power Cell (A123)
Fits model to experimental 1D, 1C, 2C, 3C, 4C voltage curves.
Uses scipy.optimize.differential_evolution for global optimization.
"""

import pybamm as pb
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
import json
import time as timer

warnings.filterwarnings("ignore")
style.use("ggplot")

# =============================================================================
# 1. Load experimental data (excluding ocv, 2D, 12D)
# =============================================================================
DATA_FILE = "A123_data_interpolated.xlsx"
OCP_FILE = "Hunan_interpolated.xlsx"

SHEETS = ["1D", "1C", "2C", "3C", "4C"]
MAX_PTS = 400  # subsample experimental data for speed

exp_data = {}
for sheet in SHEETS:
    df = pd.read_excel(DATA_FILE, sheet_name=sheet)
    step = max(1, len(df) // MAX_PTS)
    exp_data[sheet] = {
        "t": df["t"].values[::step].astype(float),
        "V": df["V"].values[::step].astype(float),
    }

# Load negative OCP lookup
df_gr = pd.read_excel(OCP_FILE)
neg_array = df_gr.to_numpy()
U_ocp = df_gr["U"].values.astype(float)
x_ocp = df_gr["x"].values.astype(float)

# =============================================================================
# 2. Positive electrode OCP (fixed, not optimized)
# =============================================================================
def LFP_ocp_Afshar2017(sto):
    c1 = -150 * sto
    c2 = -30 * (1 - sto)
    return 3.4177 - 0.020269 * sto + 0.5 * np.exp(c1) - 0.9 * np.exp(c2)

# =============================================================================
# 3. Define all optimization parameters with (name, nominal, lo, hi)
# =============================================================================
PARAM_DEFS = [
    # ---- Geometric / structural ----
    ("neg_cc_thick",       7.44e-05,   4e-05,    1.5e-04),
    ("neg_elec_thick",     3.63e-05,   1.5e-05,  8e-05),
    ("sep_thick",          2e-05,      1e-05,    5e-05),
    ("pos_elec_thick",     6.24e-05,   3e-05,    1.2e-04),
    ("pos_cc_thick",       5.75e-05,   3e-05,    1.2e-04),
    ("elec_height",        1.713,      1.0,      2.5),
    ("elec_width",         0.11,       0.06,     0.2),
    ("contact_res",        0.006,      0.0005,   0.02),
    # ---- Negative electrode ----
    ("neg_particle_radius", 5.38e-07,  1e-07,    2e-06),
    ("neg_am_vf",           0.58,      0.3,      0.75),
    ("neg_porosity",        0.36,      0.15,     0.55),
    ("neg_brugg_elyte",     1.5,       1.0,      4.0),
    ("neg_brugg_elec",      1.5,       1.0,      4.0),
    ("neg_conductivity",    107.5,     20.0,     500.0),
    # ---- Positive electrode ----
    ("pos_particle_radius", 2.85e-08,  5e-09,    2e-07),
    ("pos_conductivity",    0.036,     0.005,    1.0),
    ("pos_am_vf",           0.360529,  0.15,     0.6),
    ("pos_porosity",        0.55,      0.25,     0.7),
    ("pos_brugg_elyte",     1.5,       1.0,      4.0),
    ("pos_brugg_elec",      1.5,       1.0,      4.0),
    # ---- Separator / electrolyte / thermal ----
    ("sep_porosity",        0.45,      0.2,      0.7),
    ("sep_brugg",           1.5,       1.0,      4.0),
    ("t_plus",              0.38,      0.2,      0.6),
    ("thermo_factor",       1.0,       0.5,      2.5),
    ("htc",                 12.0,      2.0,      35.0),
    # ---- Exchange current density & activation energy ----
    ("neg_m_ref",           2.48e-7,   1e-8,     1e-5),
    ("neg_E_r",             35000,     15000,    60000),
    ("pos_m_ref",           2.0e-7,    1e-8,     1e-5),
    ("pos_E_r",             39570,     15000,    60000),
    # ---- Solid-phase diffusivity & activation energy ----
    ("neg_D_ref",           2.4e-15,   1e-16,    1e-13),
    ("neg_E_D",             35000,     15000,    60000),
    ("pos_D_ref",           1.15e-18,  1e-20,    1e-16),
    ("pos_E_D",             43000,     15000,    60000),
    # ---- Electrolyte conductivity scaling ----
    ("elyte_cond_scale",    1.0,       0.1,      5.0),
    # ---- MSMR gallery: U0 (7 reactions) ----
    ("U0_0",  0.08062545,  0.02,  0.15),
    ("U0_1",  0.11356947,  0.04,  0.22),
    ("U0_2",  0.1081655,   0.04,  0.22),
    ("U0_3",  0.11370952,  0.04,  0.25),
    ("U0_4",  0.19151176,  0.08,  0.35),
    ("U0_5",  0.31958092,  0.12,  0.55),
    ("U0_6",  0.44824024,  0.20,  0.75),
    # ---- MSMR gallery: X (fractional capacity, 7 reactions) ----
    ("X_0",   0.3838419,   0.05,  0.65),
    ("X_1",   0.13241849,  0.01,  0.40),
    ("X_2",   0.09761792,  0.01,  0.30),
    ("X_3",   0.3040794,   0.05,  0.55),
    ("X_4",   0.04173912,  0.005, 0.20),
    ("X_5",   0.01555115,  0.001, 0.10),
    ("X_6",   0.03318803,  0.005, 0.15),
    # ---- MSMR gallery: omega (widths, 7 reactions) ----
    ("w_0",   0.09545492,  0.01,  0.50),
    ("w_1",   0.07039945,  0.01,  0.50),
    ("w_2",   0.17943078,  0.02,  1.0),
    ("w_3",   1.28833751,  0.10,  5.0),
    ("w_4",   0.13084723,  0.02,  1.0),
    ("w_5",   1.31652939,  0.10,  5.0),
    ("w_6",   4.08358201,  0.50,  12.0),
    # ---- Initial SOC (stoichiometries) ----
    ("init_neg_soc_dis",  0.75,   0.55,  0.95),  # discharge start
    ("init_pos_soc_dis",  0.01,   0.005, 0.15),
    ("init_neg_soc_chg",  0.02,   0.005, 0.15),  # charge start
    ("init_pos_soc_chg",  0.95,   0.80,  0.999),
]

PARAM_NAMES = [p[0] for p in PARAM_DEFS]
NOMINAL = np.array([p[1] for p in PARAM_DEFS])
BOUNDS = [(p[2], p[3]) for p in PARAM_DEFS]
N_PARAMS = len(PARAM_DEFS)
print(f"Total parameters to optimise: {N_PARAMS}")

# =============================================================================
# 4. Build MSMR gallery OCP from parameters
# =============================================================================
GALLERY_WINDOW = np.array([0, 1.5])
F_CONST = 96485.33212
R_CONST = 8.314462618
T_REF = 298.15
f_INV_THERMAL = F_CONST / (R_CONST * T_REF)


def build_gallery_ocp(U0_arr, X_arr, w_arr):
    """Reconstruct graphite OCP from MSMR gallery parameters."""
    U = U_ocp.copy()
    x = np.zeros((len(U), 7), dtype=float)
    lo, hi = GALLERY_WINDOW
    mask = (U >= lo) & (U <= hi)
    if not np.any(mask):
        return np.zeros(len(U))
    for i in range(7):
        z = (f_INV_THERMAL * (U[mask] - U0_arr[i])) / w_arr[i]
        x[mask, i] = X_arr[i] * 0.5 * (1.0 - np.tanh(0.5 * z))
    return x.sum(axis=1)


# =============================================================================
# 5. Build PyBaMM parameter dict from optimization vector
# =============================================================================
def build_pybamm_params(p, mode="discharge"):
    """
    Build the full PyBaMM ParameterValues from an optimisation vector.
    mode: 'discharge' or 'charge' (sets initial concentrations).
    """
    d = dict(zip(PARAM_NAMES, p))

    # --- function closures capturing current parameter values ---
    _neg_m = d["neg_m_ref"]
    _neg_er = d["neg_E_r"]
    def neg_j0(c_e, c_s_surf, c_s_max, T):
        arr = np.exp(_neg_er / pb.constants.R * (1 / 298.15 - 1 / T))
        return _neg_m * arr * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

    _pos_m = d["pos_m_ref"]
    _pos_er = d["pos_E_r"]
    def pos_j0(c_e, c_s_surf, c_s_max, T):
        arr = np.exp(_pos_er / pb.constants.R * (1 / 298.15 - 1 / T))
        return _pos_m * arr * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf)**0.5

    _neg_dref = d["neg_D_ref"]
    _neg_ed = d["neg_E_D"]
    def neg_diff(sto, T):
        return _neg_dref * np.exp(_neg_ed / pb.constants.R * (1 / 298.15 - 1 / T))

    _pos_dref = d["pos_D_ref"]
    _pos_ed = d["pos_E_D"]
    def pos_diff(sto, T):
        return _pos_dref * np.exp(_pos_ed / pb.constants.R * (1 / 298.15 - 1 / T))

    _cs = d["elyte_cond_scale"]
    def elyte_cond(c_e, T):
        cm = 1e-3 * c_e
        sigma_296 = 0.2667 * cm**3 - 1.2983 * cm**2 + 1.7919 * cm + 0.1726
        E_k_e = 1.71e4
        C = 296 * np.exp(E_k_e / (pb.constants.R * 296))
        return _cs * C * sigma_296 * np.exp(-E_k_e / (pb.constants.R * T)) / T

    def elyte_diff(c_e, T):
        inputs = {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T}
        sigma_e = pb.FunctionParameter("Electrolyte conductivity [S.m-1]", inputs)
        return (pb.constants.k_b / (pb.constants.F * pb.constants.q_e)) * sigma_e * T / c_e

    # --- MSMR gallery OCP ---
    U0_arr = np.array([d[f"U0_{i}"] for i in range(7)])
    X_arr = np.array([d[f"X_{i}"] for i in range(7)])
    w_arr = np.array([d[f"w_{i}"] for i in range(7)])
    x_gr = build_gallery_ocp(U0_arr, X_arr, w_arr)

    # --- initial concentrations ---
    c_max_neg = 30985.0
    c_max_pos = 22882.0
    if mode == "discharge":
        init_neg = d["init_neg_soc_dis"] * c_max_neg
        init_pos = d["init_pos_soc_dis"] * c_max_pos
    else:
        init_neg = d["init_neg_soc_chg"] * c_max_neg
        init_pos = d["init_pos_soc_chg"] * c_max_pos

    param_dict = {
        # Cell geometry
        "Negative current collector thickness [m]": d["neg_cc_thick"],
        "Negative electrode thickness [m]": d["neg_elec_thick"],
        "Separator thickness [m]": d["sep_thick"],
        "Positive electrode thickness [m]": d["pos_elec_thick"],
        "Positive current collector thickness [m]": d["pos_cc_thick"],
        "Electrode height [m]": d["elec_height"],
        "Electrode width [m]": d["elec_width"],
        "Cell cooling surface area [m2]": 0.00533,
        "Cell volume [m3]": 3.46e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        # Current collectors
        "Negative current collector conductivity [S.m-1]": 0.75 * 58411000.0,
        "Positive current collector conductivity [S.m-1]": 0.75 * 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        # Cell
        "Nominal cell capacity [A.h]": 2.5,
        "Current function [A]": 2.5,
        "Contact resistance [Ohm]": d["contact_res"],
        # Negative electrode
        "Maximum concentration in negative electrode [mol.m-3]": c_max_neg,
        "Initial concentration in negative electrode [mol.m-3]": init_neg,
        "Negative electrode exchange-current density [A.m-2]": neg_j0,
        "Negative electrode OCP [V]": ("Gr_OCP", [np.asarray(x_gr), np.asarray(U_ocp)]),
        "Negative particle radius [m]": d["neg_particle_radius"],
        "Negative particle diffusivity [m2.s-1]": neg_diff,
        "Negative electrode active material volume fraction": d["neg_am_vf"],
        "Negative electrode porosity": d["neg_porosity"],
        "Negative electrode Bruggeman coefficient (electrolyte)": d["neg_brugg_elyte"],
        "Negative electrode Bruggeman coefficient (electrode)": d["neg_brugg_elec"],
        "Negative electrode conductivity [S.m-1]": d["neg_conductivity"],
        "Negative electrode density [kg.m-3]": 2260,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        # Positive electrode
        "Positive electrode OCP [V]": LFP_ocp_Afshar2017,
        "Maximum concentration in positive electrode [mol.m-3]": c_max_pos,
        "Initial concentration in positive electrode [mol.m-3]": init_pos,
        "Positive particle diffusivity [m2.s-1]": pos_diff,
        "Positive particle radius [m]": d["pos_particle_radius"],
        "Positive electrode conductivity [S.m-1]": d["pos_conductivity"],
        "Positive electrode active material volume fraction": d["pos_am_vf"],
        "Positive electrode porosity": d["pos_porosity"],
        "Positive electrode Bruggeman coefficient (electrolyte)": d["pos_brugg_elyte"],
        "Positive electrode Bruggeman coefficient (electrode)": d["pos_brugg_elec"],
        "Positive electrode exchange-current density [A.m-2]": pos_j0,
        "Positive electrode density [kg.m-3]": 3610,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        # Separator
        "Separator porosity": d["sep_porosity"],
        "Separator Bruggeman coefficient (electrolyte)": d["sep_brugg"],
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        # Electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1200.0,
        "Cation transference number": d["t_plus"],
        "Thermodynamic factor": d["thermo_factor"],
        "Electrolyte diffusivity [m2.s-1]": elyte_diff,
        "Electrolyte conductivity [S.m-1]": elyte_cond,
        # Experiment / thermal
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": d["htc"],
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 5.0,
        "Open-circuit voltage at 0% SOC [V]": 2.0,
        "Open-circuit voltage at 100% SOC [V]": 3.65,
        "Initial temperature [K]": 298.15,
    }
    return pb.ParameterValues(param_dict)


# =============================================================================
# 6. Experiment definitions for each C-rate
# =============================================================================
# 1D  = 1C discharge,  1C/2C/3C/4C = CC-CV charge
EXPERIMENT_DEFS = {
    "1D": {
        "mode": "discharge",
        "steps": ["Discharge at 2.5 A until 2 V"],
    },
    "1C": {
        "mode": "charge",
        "steps": [
            "Charge at 2.5 A until 3.6 V",
            "Hold at 3.6 V until 50 mA",
        ],
    },
    "2C": {
        "mode": "charge",
        "steps": [
            "Charge at 5 A until 3.6 V",
            "Hold at 3.6 V until 50 mA",
        ],
    },
    "3C": {
        "mode": "charge",
        "steps": [
            "Charge at 7.5 A until 3.6 V",
            "Hold at 3.6 V until 50 mA",
        ],
    },
    "4C": {
        "mode": "charge",
        "steps": [
            "Charge at 10 A until 3.6 V",
            "Hold at 3.6 V until 50 mA",
        ],
    },
}

# Weights per C-rate (increase weight on rates you care about most)
WEIGHTS = {"1D": 1.0, "1C": 1.0, "2C": 1.0, "3C": 1.0, "4C": 1.0}


# =============================================================================
# 7. Single-rate simulation runner
# =============================================================================
def run_single_rate(rate_key, param_vec):
    """
    Run PyBaMM simulation for one C-rate. Returns (sim_time, sim_voltage)
    or (None, None) on failure.
    """
    edef = EXPERIMENT_DEFS[rate_key]
    try:
        model = pb.lithium_ion.DFN(
            {
                "particle phases": ("1"),
                "open-circuit potential": ("single"),
                "cell geometry": "arbitrary",
                "intercalation kinetics": "symmetric Butler-Volmer",
                "diffusivity": "single",
                "thermal": "lumped",
                "contact resistance": "true",
            }
        )
        param = build_pybamm_params(param_vec, mode=edef["mode"])
        experiment = pb.Experiment([tuple(edef["steps"])])
        solver = pb.IDAKLUSolver()
        sim = pb.Simulation(
            model, parameter_values=param, experiment=experiment, solver=solver
        )
        sol = sim.solve()
        t_sim = sol["Time [s]"].entries
        V_sim = sol["Voltage [V]"].entries
        return t_sim, V_sim
    except Exception:
        return None, None


# =============================================================================
# 8. Compute RMSE between simulation and experiment for one rate
# =============================================================================
def rmse_single(rate_key, param_vec):
    """RMSE for a single C-rate. Returns large penalty on failure."""
    t_sim, V_sim = run_single_rate(rate_key, param_vec)
    if t_sim is None:
        return 1.0  # penalty for failed simulation

    t_exp = exp_data[rate_key]["t"]
    V_exp = exp_data[rate_key]["V"]

    # Interpolate simulation onto experimental time grid
    # Only compare within the overlapping time range
    t_max = min(t_sim[-1], t_exp[-1])
    t_min = max(t_sim[0], t_exp[0])
    mask = (t_exp >= t_min) & (t_exp <= t_max)
    if mask.sum() < 5:
        return 1.0

    try:
        f_interp = interp1d(t_sim, V_sim, kind="linear", fill_value="extrapolate")
        V_sim_interp = f_interp(t_exp[mask])
        rmse = np.sqrt(np.mean((V_sim_interp - V_exp[mask]) ** 2))
        return rmse
    except Exception:
        return 1.0


# =============================================================================
# 9. Total objective function
# =============================================================================
eval_count = 0
best_cost = np.inf
best_params = None


def objective(param_vec):
    """Weighted sum of RMSE across all C-rates."""
    global eval_count, best_cost, best_params
    eval_count += 1

    total = 0.0
    details = {}
    for rate in SHEETS:
        r = rmse_single(rate, param_vec)
        details[rate] = r
        total += WEIGHTS[rate] * r

    if total < best_cost:
        best_cost = total
        best_params = param_vec.copy()
        detail_str = "  ".join(f"{k}:{v:.4f}" for k, v in details.items())
        print(f"[Eval {eval_count:>5d}]  Cost={total:.5f}  ({detail_str})")

    return total


# =============================================================================
# 10. Run optimisation
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DFN Parameter Optimiser  --  LFP High-Power Cell (A123)")
    print(f"Fitting rates: {SHEETS}")
    print(f"Parameters: {N_PARAMS}")
    print("=" * 70)

    t0 = timer.time()
    p_best = NOMINAL.copy()
    cost_best = np.inf

    try:
        # --- Phase 1: Local optimisation (Powell) ---
        print("\n>>> Phase 1: Local optimisation (Powell) from nominal values ...")
        print("    (Ctrl+C at any time to stop and save best result)")
        res_local = minimize(
            objective,
            NOMINAL,
            method="Powell",
            bounds=BOUNDS,
            options={"maxiter": 200, "maxfev": 100, "ftol": 1e-5},
        )
        t1 = timer.time()
        print(f"Phase 1 done in {(t1 - t0)/60:.1f} min  |  cost = {res_local.fun:.5f}")
        if res_local.fun < cost_best:
            p_best = res_local.x.copy()
            cost_best = res_local.fun
        if best_params is not None and best_cost < cost_best:
            p_best = best_params.copy()
            cost_best = best_cost

        # --- Phase 2: Global refinement with differential evolution ---
        print("\n>>> Phase 2: Global optimisation (differential_evolution) ...")
        print("    (Ctrl+C at any time to stop and save best result)")
        res_global = differential_evolution(
            objective,
            bounds=BOUNDS,
            x0=p_best,
            maxiter=80,
            popsize=8,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            updating="deferred",
            workers=1,
            disp=True,
        )
        if res_global.fun < cost_best:
            p_best = res_global.x.copy()
            cost_best = res_global.fun

    except KeyboardInterrupt:
        print("\n\n>>> Interrupted! Saving best result found so far ...")
        if best_params is not None and best_cost < cost_best:
            p_best = best_params.copy()
            cost_best = best_cost

    t2 = timer.time()
    print(f"\nOptimisation ran for {(t2 - t0)/60:.1f} min  |  {eval_count} evaluations")
    print(f"Best cost = {cost_best:.5f}")

    # =================================================================
    # 11. Save optimised parameters
    # =================================================================
    opt_dict = dict(zip(PARAM_NAMES, p_best.tolist()))
    opt_dict["__cost__"] = float(cost_best)
    opt_dict["__eval_count__"] = eval_count
    with open("optimised_params.json", "w") as f:
        json.dump(opt_dict, f, indent=2)
    print("Parameters saved to optimised_params.json")

    # =================================================================
    # 12. Plot comparison: experiment vs optimised model
    # =================================================================
    fig, axes = plt.subplots(1, len(SHEETS), figsize=(5 * len(SHEETS), 4), sharey=True)
    if len(SHEETS) == 1:
        axes = [axes]

    for ax, rate in zip(axes, SHEETS):
        # Experimental
        ax.plot(exp_data[rate]["t"], exp_data[rate]["V"], "k-", lw=1.5, label="Exp")

        # Simulated with optimised params
        t_sim, V_sim = run_single_rate(rate, p_best)
        if t_sim is not None:
            ax.plot(t_sim, V_sim, "r--", lw=1.2, label="Model (opt)")

        # Simulated with nominal params
        t_nom, V_nom = run_single_rate(rate, NOMINAL)
        if t_nom is not None:
            ax.plot(t_nom, V_nom, "b:", lw=1.0, alpha=0.6, label="Model (nom)")

        ax.set_title(rate)
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Voltage [V]")
    plt.tight_layout()
    plt.savefig("optimisation_fit.png", dpi=200)
    plt.show()
    print("Plot saved to optimisation_fit.png")

    # =================================================================
    # 13. Print per-rate RMSE summary
    # =================================================================
    print("\n" + "=" * 50)
    print("Per-rate RMSE (optimised):")
    for rate in SHEETS:
        r = rmse_single(rate, p_best)
        print(f"  {rate}: {r*1000:.2f} mV")
    print("=" * 50)
