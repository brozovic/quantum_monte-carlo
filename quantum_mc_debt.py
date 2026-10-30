"""
Quantum Monte Carlo (Quantum Amplitude Estimation) for the public-debt model.

Goal : estimate  P( d_T > threshold )  = probability that the debt/GDP ratio at
       the horizon exceeds a threshold (a tail-risk number).

Pipeline (see the explanation that goes with this file):
  1. Reduce the model to 2 aggregate random factors (growth shock G, rate shock R)
  2. Linearise d_T(G,R)  ->  d_T ~ d_const + c1*x1 + c2*x2   (x1,x2 = integers)
  3. Load the two Gaussian distributions into qubits         (state preparation)
  4. Compute d_T with a quantum adder                         (WeightedAdder)
  5. Flag  d_T > threshold into one "objective" qubit         (IntegerComparator)
  6. Estimate P(objective = 1) with amplitude estimation      (Grover operator + MLAE)

Requires:  pip install qiskit numpy scipy
If qiskit is missing, the script runs an ideal (noise-free, analytic) emulation of
the amplitude-estimation step so that you can still test the whole workflow.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. PARAMETERS  (same economics as the classical model)
# ----------------------------------------------------------------------
@dataclass
class P:
    T: int = 20
    d0: float = 0.90
    d_star: float = 0.60
    b: float = 0.03
    a: float = 0.0
    g_bar: float = 0.015
    r_bar: float = 0.025
    sigma_g: float = 0.010          # std of yearly growth shock
    sigma_r: float = 0.005          # std of yearly interest-rate shock
    pb_policy: float = 0.0          # constant yearly policy effect on pb (% GDP)

    n_bits: int = 5                 # qubits per random factor (2^n grid points)
    n_sigma: float = 3.0            # truncation of the Gaussians (+/- n_sigma)
    sum_bits: int = 6               # qubits of the quantum adder (resolution of d_T)
    threshold: float = 0.95         # event: d_T > 95 % of GDP

    # amplitude estimation (MLAE) settings
    grover_powers: tuple = (0, 1, 2, 4)
    shots: int = 500
    seed: int = 1


# ----------------------------------------------------------------------
# 1. REDUCED CLASSICAL MODEL:  d_T as a function of two aggregate shocks
# ----------------------------------------------------------------------
def debt_T(G, R, p: P):
    """G = cumulative growth shock, R = cumulative rate shock over T years.
    Shocks are spread evenly over the years. Deterministic path otherwise."""
    d = p.d0
    g = p.g_bar + G / p.T
    r = p.r_bar + R / p.T
    for _ in range(p.T):
        pb = p.a + p.b * (d - p.d_star) + p.pb_policy
        lam = (r - g) / (1 + g)
        d = (1 + lam) * d - pb
    return d


# ----------------------------------------------------------------------
# 2. DISCRETISE + LINEARISE  ->  integer arithmetic the circuit can do
# ----------------------------------------------------------------------
def build_problem(p: P):
    n = p.n_bits
    M = 2 ** n
    sdG = p.sigma_g * math.sqrt(p.T)       # std of cumulative growth shock
    sdR = p.sigma_r * math.sqrt(p.T)       # std of cumulative rate shock

    # Both variables are defined as ADVERSE shocks so their effect on debt is >= 0
    #   u1 = -G  (low growth is bad),  u2 = +R (high rates are bad)
    lo1, hi1 = -p.n_sigma * sdG, p.n_sigma * sdG
    lo2, hi2 = -p.n_sigma * sdR, p.n_sigma * sdR
    grid1 = np.linspace(lo1, hi1, M)
    grid2 = np.linspace(lo2, hi2, M)

    # discrete Gaussian probabilities (renormalised after truncation)
    p1 = norm.pdf(grid1, 0, sdG); p1 /= p1.sum()
    p2 = norm.pdf(grid2, 0, sdR); p2 /= p2.sum()

    # linearisation by finite differences around (0,0)
    h1, h2 = sdG * 0.5, sdR * 0.5
    c1_per_unit = (debt_T(-h1, 0, p) - debt_T(h1, 0, p)) / (2 * h1)   # dd/du1 (>0)
    c2_per_unit = (debt_T(0, h2, p) - debt_T(0, -h2, p)) / (2 * h2)   # dd/du2 (>0)
    step1 = grid1[1] - grid1[0]
    step2 = grid2[1] - grid2[0]
    c1, c2 = c1_per_unit * step1, c2_per_unit * step2     # d increase per grid step
    # linear model: d(x1,x2) = d0_lin + c1*x1 + c2*x2   (d0_lin = intercept at x1=x2=0)
    d0_lin = debt_T(0, 0, p) - c1 * (M - 1) / 2 - c2 * (M - 1) / 2

    # integer resolution of the adder: total range of d spread over 2^sum_bits - 1
    d_range = (c1 + c2) * (M - 1)
    delta = d_range / (2 ** p.sum_bits - 1)
    w1 = [int(round(c1 * 2 ** j / delta)) for j in range(n)]   # per-bit weights x1
    w2 = [int(round(c2 * 2 ** j / delta)) for j in range(n)]   # per-bit weights x2
    # integer sum S(x1,x2) = sum_j w1[j]*bit_j(x1) + sum_j w2[j]*bit_j(x2)
    # d ~ d0_lin + delta * S
    thr_int = max(0, math.ceil((p.threshold - d0_lin) / delta))   # flag S >= thr_int

    return dict(grid1=grid1, grid2=grid2, p1=p1, p2=p2, w1=w1, w2=w2,
                delta=delta, d0_lin=d0_lin, thr_int=thr_int, n=n, M=M,
                sdG=sdG, sdR=sdR, c1=c1, c2=c2, d_const=d0_lin,
                c1_per_unit=c1_per_unit, c2_per_unit=c2_per_unit,
                step1=step1, step2=step2, lo1=lo1, hi1=hi1, lo2=lo2, hi2=hi2)


def int_sum(x1, x2, pr):
    s = 0
    for j, w in enumerate(pr["w1"]):
        s += w * ((x1 >> j) & 1)
    for j, w in enumerate(pr["w2"]):
        s += w * ((x2 >> j) & 1)
    return s


def reference_probabilities(p: P, pr):
    """Classical reference values for the same event."""
    M = pr["M"]
    # (a) exactly what the circuit computes (discretised + integer arithmetic)
    a_circuit = sum(pr["p1"][i] * pr["p2"][j]
                    for i in range(M) for j in range(M)
                    if int_sum(i, j, pr) >= pr["thr_int"])
    # (b) discretised grid but exact non-linear model
    a_grid = sum(pr["p1"][i] * pr["p2"][j]
                 for i in range(M) for j in range(M)
                 if debt_T(-pr["grid1"][i], pr["grid2"][j], p) > p.threshold)
    # (c) continuous Gaussian, exact non-linear model (big classical MC)
    rng = np.random.default_rng(123)
    G = rng.normal(0, pr["sdG"], 2_000_000)
    R = rng.normal(0, pr["sdR"], 2_000_000)
    # vectorised copy of debt_T
    d = np.full_like(G, p.d0); g = p.g_bar + G / p.T; r = p.r_bar + R / p.T
    for _ in range(p.T):
        pb = p.a + p.b * (d - p.d_star) + p.pb_policy
        d = (1 + (r - g) / (1 + g)) * d - pb
    a_cont = (d > p.threshold).mean()
    return a_circuit, a_grid, a_cont


# ----------------------------------------------------------------------
# 3. QISKIT CIRCUIT  A |0> = sqrt(1-a)|bad,0> + sqrt(a)|good,1>
# ----------------------------------------------------------------------
def build_A(pr):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, WeightedAdder, IntegerComparator

    n = pr["n"]
    weights = pr["w1"] + pr["w2"]
    n_state = 2 * n
    adder = WeightedAdder(n_state, weights)            # qubits: state | sum | carry
    n_sum = adder.num_sum_qubits
    n_adder_anc = adder.num_qubits - n_state - n_sum

    thr = min(pr["thr_int"], 2 ** n_sum)               # safety clamp
    comp = IntegerComparator(n_sum, thr, geq=True)     # qubits: sum | flag | ancillas
    n_comp_anc = comp.num_qubits - n_sum - 1

    # global qubit layout
    x1 = list(range(0, n))
    x2 = list(range(n, 2 * n))
    ssum = list(range(n_state, n_state + n_sum))
    anc_a = list(range(n_state + n_sum, n_state + n_sum + n_adder_anc))
    obj = n_state + n_sum + n_adder_anc
    anc_c = list(range(obj + 1, obj + 1 + n_comp_anc))
    total = obj + 1 + n_comp_anc

    A = QuantumCircuit(total, name="A")
    A.append(StatePreparation(np.sqrt(pr["p1"])), x1)           # load distribution 1
    A.append(StatePreparation(np.sqrt(pr["p2"])), x2)           # load distribution 2
    A.append(adder.to_gate(), x1 + x2 + ssum + anc_a)           # d_T (integer) -> sum reg
    A.append(comp.to_gate(), ssum + [obj] + anc_c)              # flag d_T > threshold
    return A, obj, total


def make_grover_prob_fn(pr):
    """Returns f(k) = P(objective=1) after A followed by k Grover iterations."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator
    from qiskit.quantum_info import Statevector

    A, obj, total = build_A(pr)
    oracle = QuantumCircuit(total, name="S_good")
    oracle.z(obj)                                       # phase flip on 'good' states
    Q = grover_operator(oracle, state_preparation=A)    # Q = A S_0 A^dag S_good
    print(f"Circuit: {total} qubits | A depth ~ {A.decompose(reps=3).depth()}")

    cache = {}
    def prob(k):
        if k not in cache:
            qc = QuantumCircuit(total)
            qc.compose(A, inplace=True)
            for _ in range(k):
                qc.compose(Q, inplace=True)
            sv = Statevector(qc)                        # use Aer / hardware for big runs
            cache[k] = float(sv.probabilities([obj])[1])
        return cache[k]
    return prob


def make_ideal_prob_fn(a_true):
    """Noise-free emulation: P_k = sin^2((2k+1) theta_a)."""
    th = math.asin(math.sqrt(a_true))
    return lambda k: math.sin((2 * k + 1) * th) ** 2


# ----------------------------------------------------------------------
# 4. MAXIMUM-LIKELIHOOD AMPLITUDE ESTIMATION (MLAE)
# ----------------------------------------------------------------------
def mlae(prob_fn, powers, shots, seed=1):
    rng = np.random.default_rng(seed)
    hits = {k: rng.binomial(shots, prob_fn(k)) for k in powers}    # "measurements"
    thetas = np.linspace(1e-6, math.pi / 2 - 1e-6, 200_001)
    ll = np.zeros_like(thetas)
    for k, h in hits.items():
        s2 = np.sin((2 * k + 1) * thetas) ** 2
        s2 = np.clip(s2, 1e-12, 1 - 1e-12)
        ll += h * np.log(s2) + (shots - h) * np.log(1 - s2)
    th = thetas[np.argmax(ll)]
    a_hat = math.sin(th) ** 2
    queries = sum(shots * (2 * k + 1) for k in powers)             # calls to A
    return a_hat, queries


# ----------------------------------------------------------------------
# 4b. GRAPHS OF THE EXPERIMENT RESULTS
# ----------------------------------------------------------------------
def plot_results(p: P, pr, refs, prob_fn, a_hat, backend,
                 filename="quantum_results.png", show=True):
    """Draw 4 panels:
       A. comparison of the probabilities (classical vs circuit vs QAE)
       B. distribution of d_T seen by the circuit, tail in red
       C. Grover amplification curve P(flag=1) vs number of iterations k
       D. estimation error vs cost: QAE vs classical Monte Carlo
    refs = (a_circuit, a_grid, a_cont) from reference_probabilities()."""
    a_circuit, a_grid, a_cont = refs
    M = pr["M"]

    # --- data for B: debt level and probability of each of the M*M grid points
    d_vals = np.array([pr["d0_lin"] + pr["delta"] * int_sum(i, j, pr)
                       for i in range(M) for j in range(M)])
    d_prob = np.array([pr["p1"][i] * pr["p2"][j]
                       for i in range(M) for j in range(M)])
    order = np.argsort(d_vals)
    dv, dp = d_vals[order], d_prob[order]

    # --- data for C: Grover curve (circuit points + theory line)
    ks = np.arange(0, 6)
    p_k = np.array([prob_fn(int(k)) for k in ks])
    theta = math.asin(math.sqrt(a_circuit))

    # --- data for D: error vs calls (ideal Grover probabilities, 200 repetitions)
    ideal = make_ideal_prob_fn(a_circuit)
    schedules = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 4, 8), (0, 1, 2, 4, 8, 16)]
    q_calls, q_rmse, q_se = [], [], []
    for sch in schedules:
        est = [mlae(ideal, sch, p.shots, seed=sd)[0] for sd in range(200)]
        q_calls.append(sum(p.shots * (2 * k + 1) for k in sch))
        q_rmse.append(math.sqrt(np.mean((np.array(est) - a_circuit) ** 2)))
        q_se.append(q_rmse[-1] / math.sqrt(2 * (len(est) - 1)))   # approx. s.e. of an RMSE
    N = np.logspace(2, 5.5, 50)
    c_err = np.sqrt(a_circuit * (1 - a_circuit) / N)

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # A
    labels = ["Continuous\n(exact model)", "Discretised\n(exact model)",
              "Circuit target\n(linear+integer)", f"QAE estimate\n({backend})"]
    vals = [a_cont, a_grid, a_circuit, a_hat]
    bars = ax[0, 0].bar(labels, vals, color=["#888", "#aaa", "tab:blue", "tab:orange"])
    for b, v in zip(bars, vals):
        ax[0, 0].text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    ax[0, 0].set_title(f"A. P(debt/GDP > {p.threshold:.0%} at year {p.T})")
    ax[0, 0].set_ylabel("probability")

    # B
    colors = ["tab:red" if v > p.threshold else "tab:blue" for v in dv]
    ax[0, 1].bar(range(len(dv)), dp, color=colors, width=1.0)
    ticks = np.linspace(0, len(dv) - 1, 6).astype(int)
    ax[0, 1].set_xticks(ticks)
    ax[0, 1].set_xticklabels([f"{dv[t]:.0%}" for t in ticks])
    ax[0, 1].set_xlabel("debt/GDP at horizon (grid points, sorted)")
    ax[0, 1].set_ylabel("probability of grid point")
    ax[0, 1].set_title(f"B. Distribution seen by the circuit (red = above {p.threshold:.0%})")

    # C
    kk = np.linspace(0, 5, 400)
    ax[1, 0].plot(kk, np.sin((2 * kk + 1) * theta) ** 2, color="tab:blue", alpha=0.6,
                  label=r"$\sin^2((2k+1)\theta)$")
    ax[1, 0].scatter(ks, p_k, color="tab:red", zorder=3, label=f"circuit ({backend})")
    ax[1, 0].axhline(a_circuit, color="gray", ls="--", label=f"a = {a_circuit:.3f}")
    ax[1, 0].set_xlabel("Grover iterations k")
    ax[1, 0].set_ylabel("P(objective qubit = 1)")
    ax[1, 0].set_title("C. Amplitude amplification")
    ax[1, 0].legend()

    # D
    ax[1, 1].loglog(N, c_err, color="tab:gray", lw=2, ls="--", label="Classical MC ~ 1/sqrt(N)")
    ax[1, 1].errorbar(q_calls, q_rmse, yerr=q_se, fmt="o-", color="tab:orange", lw=2,
                      capsize=3, label="MLAE (ideal emulation, 200 reps, +/-1 s.e.)")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log"); ax[1, 1].set_xlim(300, 5e5)
    ax[1, 1].set_xlabel("calls to the circuit A / classical samples")
    ax[1, 1].set_ylabel("RMSE of the probability estimate")
    ax[1, 1].set_title("D. Error vs cost")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("Quantum Monte Carlo for debt risk", fontsize=14)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    print(f"Graphs saved to {filename}")
    import os
    os.makedirs("images", exist_ok=True)
    fig.canvas.draw()
    for axis, name in zip(ax.flatten(), "ABCD"):
        ext = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"images/QMC-{name}.png", dpi=200, bbox_inches=ext.expanded(1.03, 1.06))
    print("Panels saved to images/QMC-A.png ... QMC-D.png")
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = P()
    pr = build_problem(p)
    a_circuit, a_grid, a_cont = reference_probabilities(p, pr)

    print("Integer weights x1:", pr["w1"], " x2:", pr["w2"],
          " threshold integer:", pr["thr_int"])
    print(f"\nReference  P(d_T > {p.threshold:.0%}):")
    print(f"  continuous Gaussians, exact model (2M classical samples): {a_cont:.4f}")
    print(f"  discretised grid, exact model                          : {a_grid:.4f}")
    print(f"  discretised grid + linear + integer (= what circuit does): {a_circuit:.4f}")

    try:
        prob_fn = make_grover_prob_fn(pr)
        backend = "Qiskit statevector"
    except ImportError:
        print("\n[qiskit not installed -> using ideal analytic emulation of the circuit]")
        prob_fn = make_ideal_prob_fn(a_circuit)
        backend = "ideal emulation"

    a_hat, queries = mlae(prob_fn, p.grover_powers, p.shots, p.seed)
    err = abs(a_hat - a_circuit)
    print(f"\nQAE ({backend}): estimate = {a_hat:.4f}  (error vs circuit target {err:.4f})")
    print(f"  total calls to A: {queries}")
    eps = 0.005   # target accuracy for the scaling comparison
    print(f"  for accuracy eps={eps}: classical MC ~ a(1-a)/eps^2 = "
          f"{a_circuit * (1 - a_circuit) / eps ** 2:,.0f} samples, "
          f"QAE ~ pi/(4 eps) = {math.pi / (4 * eps):,.0f} calls (up to constants/log factors)")

    # ---- graphs ----
    plot_results(p, pr, (a_circuit, a_grid, a_cont), prob_fn, a_hat, backend)
