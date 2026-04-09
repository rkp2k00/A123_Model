"""
MSMR Optimizer — 7-Gallery Graphite OCP Fit
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_EXCEL = "Hunan_interpolated.xlsx"
N_GALLERIES = 7
MAX_ITERATIONS = 25              # max convergence loops (same as RC model)
CONVERGENCE_THRESHOLD = 1e-10    # stop if cost change < this
SUMX_PENALTY_WEIGHT = 500.0      # penalty weight for sum(X)=1 constraint

# Constants
F, R, T = 96485.33212, 8.314462618, 298.15
f = F / (R * T)  # ~38.94 V^-1

# Gallery window — extended to 0.0 to cover all experimental data (U min = 0.0047)
GALLERY_WINDOW = np.array([0.0, 3.5])


# =============================================================================
# BOUNDS (per-gallery, U0 descending order)
# =============================================================================

BOUNDS_U0 = [
    (0.35, 0.70),    # G1 — highest voltage
    (0.25, 0.50),    # G2
    (0.15, 0.35),    # G3
    (0.10, 0.25),    # G4
    (0.08, 0.18),    # G5
    (0.07, 0.16),    # G6
    (0.04, 0.12),    # G7 — lowest voltage
]

BOUNDS_X = [
    (0.005, 0.06),   # G1
    (0.005, 0.06),   # G2
    (0.02,  0.15),   # G3
    (0.05,  0.35),   # G4
    (0.03,  0.20),   # G5
    (0.05,  0.25),   # G6
    (0.15,  0.50),   # G7
]

BOUNDS_W = [
    (2.0,   8.0),    # G1
    (0.5,   4.0),    # G2
    (0.001, 1.0),    # G3
    (0.3,   3.0),    # G4
    (0.05,  0.50),   # G5
    (0.01,  0.20),   # G6
    (0.01,  0.15),   # G7
]

N_RESTARTS = 30  # multi-start random restarts to escape local minima


def get_bounds():
    """Build (min, max) bounds for every parameter."""
    return BOUNDS_U0 + BOUNDS_X + BOUNDS_W


# =============================================================================
# LOAD EXPERIMENTAL DATA
# =============================================================================

logger.info(f"Loading experimental data: {INPUT_EXCEL}")
df = pd.read_excel(INPUT_EXCEL)
U_exp = df['U'].to_numpy()
x_exp = df['x'].to_numpy()
logger.info(f"Loaded {len(U_exp):,} pts | U: [{U_exp.min():.4f}, {U_exp.max():.4f}] V | x: [{x_exp.min():.4f}, {x_exp.max():.4f}]")


# =============================================================================
# MSMR FORWARD MODEL
# =============================================================================

def x_gallery(U, U0, X, omega):
    """Single gallery occupancy with gallery window clipping."""
    lo, hi = GALLERY_WINDOW[0], GALLERY_WINDOW[-1]
    mask = (U >= lo) & (U <= hi)
    out = np.zeros_like(U, dtype=float)
    if np.any(mask):
        z = (f * (U[mask] - U0)) / max(omega, 1e-12)
        out[mask] = X * 0.5 * (1.0 - np.tanh(0.5 * z))
    return out


def msmr_x_total(U, U0, X, omega):
    """Sum of all gallery contributions → total x(U)."""
    x = np.zeros_like(U, dtype=float)
    for j in range(len(U0)):
        x += x_gallery(U, U0[j], X[j], omega[j])
    return x


# =============================================================================
# PACK / UNPACK
# =============================================================================

def pack(U0, X, w):
    return np.concatenate([U0, X, w])


def unpack(theta):
    n = N_GALLERIES
    return theta[0:n], theta[n:2*n], theta[2*n:3*n]


# =============================================================================
# OBJECTIVE FUNCTION  (mirrors rc_model_standalone.py objective_function)
# =============================================================================

def objective_function(params, U_data, x_data):
    """
    Loss = 0.7*RMSE + 0.3*MAE + sum(X)=1 penalty.

    Note: RC model uses L1/L2 smoothness on 51-bin SOC curves. Here we have
    only 7 independent gallery parameters (not a curve), so smoothness reg
    is replaced by the sum(X)=1 stoichiometry constraint as a penalty.

    Parameter vector layout (N = N_GALLERIES = 7):
      [U0(N), X(N), w(N)]
    """
    U0, X, w = unpack(params)

    x_model = msmr_x_total(U_data, U0, X, w)

    err  = x_data - x_model
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    error = 0.7 * rmse + 0.3 * mae

    # sum(X) = 1 constraint penalty (replaces smoothness reg for this problem)
    penalty = SUMX_PENALTY_WEIGHT * (np.sum(X) - 1.0) ** 2

    return error + penalty


# =============================================================================
# VOLTAGE RMSE (for reporting)
# =============================================================================

def compute_voltage_rmse(U_data, x_data, x_model):
    """Approximate voltage RMSE by interpolating model U(x) curve."""
    sort_idx = np.argsort(x_model)
    x_s, U_s = x_model[sort_idx], U_data[sort_idx]
    unique_m = np.diff(x_s, prepend=-1) > 1e-12
    x_u, U_u = x_s[unique_m], U_s[unique_m]
    if len(x_u) < 10:
        return np.nan
    fn = interp1d(x_u, U_u, kind='linear', bounds_error=False, fill_value='extrapolate')
    U_at_xexp = fn(x_data)
    return np.sqrt(np.mean((U_at_xexp - U_data) ** 2))


# =============================================================================
# OPTIMIZE  (mirrors rc_model_standalone.py optimize_cycle)
# =============================================================================

def _run_one_start(U_data, x_data, U0_g, X_g, w_g, bounds, start_label=""):
    """
    Single L-BFGS-B run with iterative warm-restart convergence loop.
    Same strategy as rc_model_standalone.py optimize_cycle().
    """
    prev_cost = np.inf

    for iteration in range(MAX_ITERATIONS):

        x0 = pack(U0_g, X_g, w_g)

        result = minimize(
            objective_function, x0,
            args=(U_data, x_data),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 50000}
        )

        cost_new  = result.fun
        cost_diff = abs(prev_cost - cost_new)

        U0_opt, X_opt, w_opt = unpack(result.x)
        x_model = msmr_x_total(U_data, U0_opt, X_opt, w_opt)
        rmse = float(np.sqrt(np.mean((x_model - x_data) ** 2)))
        mae  = float(np.mean(np.abs(x_model - x_data)))
        v_rmse = compute_voltage_rmse(U_data, x_data, x_model)

        logger.info(
            f"  {start_label} Iter {iteration+1:2d}: cost {prev_cost:.6e} -> {cost_new:.6e}  "
            f"(diff={cost_diff:.4e}) | RMSE={rmse:.6e} | MAE={mae:.6e} | "
            f"V-RMSE={v_rmse*1000:.2f} mV | sum(X)={X_opt.sum():.5f}"
        )

        if cost_diff < CONVERGENCE_THRESHOLD:
            logger.info(f"  {start_label} Converged at iteration {iteration+1}.")
            break

        prev_cost = cost_new
        U0_g = U0_opt.copy()
        X_g  = X_opt.copy()
        w_g  = w_opt.copy()

    return result


def optimize_msmr(U_data, x_data):
    """
    Multi-start L-BFGS-B: run from midpoint + N_RESTARTS random starts,
    keep the best result. Each start uses the iterative warm-restart
    convergence loop from rc_model_standalone.py.
    """
    bounds = get_bounds()
    rng = np.random.default_rng(42)

    all_bounds = BOUNDS_U0 + BOUNDS_X + BOUNDS_W
    lo = np.array([b[0] for b in all_bounds])
    hi = np.array([b[1] for b in all_bounds])

    # Build starting points: midpoint first, then random within bounds
    starts = []
    mid = 0.5 * (lo + hi)
    starts.append(("mid", mid))
    for i in range(N_RESTARTS):
        theta = rng.uniform(lo, hi)
        starts.append((f"rng{i+1:02d}", theta))

    logger.info(f"Optimizing ({len(U_data):,} pts, {N_GALLERIES} galleries, {3*N_GALLERIES} params, {len(starts)} starts)...")

    best_result = None
    best_cost = np.inf

    for label, theta in starts:
        U0_g, X_g, w_g = unpack(theta)
        result = _run_one_start(U_data, x_data, U0_g, X_g, w_g, bounds, start_label=f"[{label}]")

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result
            logger.info(f"  *** New best: {label} cost={best_cost:.6e}")

    # ── Final metrics ───────────────────────────────────────────────
    U0_opt, X_opt, w_opt = unpack(best_result.x)
    x_fit = msmr_x_total(U_data, U0_opt, X_opt, w_opt)
    rmse = float(np.sqrt(np.mean((x_fit - x_data) ** 2)))
    mae  = float(np.mean(np.abs(x_fit - x_data)))
    v_rmse = compute_voltage_rmse(U_data, x_data, x_fit)

    logger.info(f"  Best: RMSE={rmse:.6e} | MAE={mae:.6e} | V-RMSE={v_rmse*1000:.2f} mV | sum(X)={X_opt.sum():.5f}")

    return {
        'U0': U0_opt, 'X': X_opt, 'w': w_opt,
        'x_fit': x_fit,
        'rmse': rmse, 'mae': mae, 'v_rmse': v_rmse,
        'cost': best_result.fun,
        'success': best_result.success, 'nit': getattr(best_result, 'nit', None),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── Run optimisation ────────────────────────────────────────────
    res = optimize_msmr(U_exp, x_exp)

    U0_opt, X_opt, w_opt = res['U0'], res['X'], res['w']
    x_fit = res['x_fit']

    # ── Print summary (same style as rc_model_standalone.py) ────────
    print("\n" + "=" * 70)
    print("  MSMR Optimisation Results — 7 Gallery Graphite")
    print("=" * 70)
    print(f"{'Gallery':>8}  {'U0 (V)':>10}  {'X':>10}  {'omega':>10}")
    print("-" * 70)
    for j in range(N_GALLERIES):
        print(f"{'G'+str(j+1):>8}  {U0_opt[j]:10.5f}  {X_opt[j]:10.5f}  {w_opt[j]:10.5f}")
    print("-" * 70)
    print(f"{'sum(X)':>8}  {'':>10}  {np.sum(X_opt):10.5f}")
    print(f"\n  Cost        = {res['cost']:.6e}")
    print(f"  x-RMSE      = {res['rmse']:.6e}")
    print(f"  x-MAE       = {res['mae']:.6e}")
    print(f"  Error (0.7R+0.3M) = {0.7*res['rmse'] + 0.3*res['mae']:.6e}")
    print(f"  Voltage RMSE      = {res['v_rmse']*1000:.2f} mV")
    print(f"  Converged   = {res['success']}")
    print(f"  Iterations  = {res['nit']}")
    print("=" * 70)

    # ── Copyable arrays ────────────────────────────────────────────
    print("\n# Copyable parameter arrays:")
    print(f"U0 = np.array({np.array2string(U0_opt, separator=', ', precision=5)})")
    print(f"X  = np.array({np.array2string(X_opt, separator=', ', precision=5)})")
    print(f"w  = np.array({np.array2string(w_opt, separator=', ', precision=5)})")

    # ── Plot 1: Overall fit ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(x_exp, U_exp, 'k-', lw=2, label='Experimental')
    ax1.plot(x_fit, U_exp, 'r--', lw=2, label='MSMR fit')
    ax1.set_xlabel('x (stoichiometry)')
    ax1.set_ylabel('U (V)')
    ax1.set_ylim([0, 1])
    ax1.set_title(f'MSMR Fit | RMSE={res["rmse"]:.4e} | V-RMSE={res["v_rmse"]*1000:.1f} mV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Individual gallery contributions ────────────────────
    ax2 = axes[1]
    for j in range(N_GALLERIES):
        xj = x_gallery(U_exp, U0_opt[j], X_opt[j], w_opt[j])
        ax2.plot(xj, U_exp, label=f'G{j+1} (U0={U0_opt[j]:.3f})')
    ax2.set_xlabel('xj')
    ax2.set_ylabel('U (V)')
    ax2.set_ylim([0, 1])
    ax2.set_xlim([-0.01, 0.5])
    ax2.set_title('Individual Gallery Contributions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ── Plot 3: Residual (same layout as RC model voltage plots) ────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(x_exp, x_exp, 'k-', lw=1.2, label='Experimental x')
    ax1.plot(x_exp, x_fit, 'r--', lw=1.0, label='Model x')
    ax1.set_ylabel('x (stoichiometry)')
    ax1.set_title(f'MSMR Fit | RMSE={res["rmse"]:.4e} | V-RMSE={res["v_rmse"]*1000:.2f} mV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_exp, (x_fit - x_exp), 'b-', lw=1.0)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_ylabel('Residual (x_model - x_exp)')
    ax2.set_xlabel('x (experimental)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
