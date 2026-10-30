"""
Regenerates every number quoted in the paper and writes
    paper_tables.md      (human-readable tables)
    paper_values.json    (all values, used by build_paper.py to fill the LaTeX)
Configuration = defaults of P() in quantum_mc_debt.py
(n_bits=5, sum_bits=6, shots=500, powers (0,1,2,4), threshold 0.95).

Run:  python paper_tables.py
Optional parts (Grover check, transpiled gate count, qubit count) run only if
Qiskit is installed; otherwise they are reported as "not run".
"""
import json, math, platform, os, argparse, time
import numpy as np
import scipy
from scipy.stats import norm, truncnorm
from quantum_mc_debt import (P, build_problem, debt_T, int_sum,
                             reference_probabilities, make_ideal_prob_fn, mlae)


ap = argparse.ArgumentParser(description="Regenerate all paper numbers")
ap.add_argument("--qiskit", action="store_true", help="ALSO run the circuit simulation (off by default)")
ap.add_argument("--skip-qiskit", action="store_true", help="(kept for compatibility; Qiskit part is off unless --qiskit)")
ap.add_argument("--kmax", type=int, default=0, help="largest Grover power in the circuit check (0 = validate A only)")
ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"], help="Aer device")
ap.add_argument("--precision", default="double", choices=["single", "double"],
                help="single halves memory/time (RTX 2060: 6 GB)")
ap.add_argument("--gates", action="store_true", help="also transpile A to {cx,rz,sx,x} (slow; run once)")
ap.add_argument("--force-qiskit", action="store_true", help="ignore qiskit_cache.json")
args, _ = ap.parse_known_args()
p = P()
pr = build_problem(p)
M, n = pr["M"], pr["n"]
V = {}                                   # all values for the paper
out = []
def w(s=""):
    print(s); out.append(s)

# ---------------- seeds / settings ----------------
SEED_REF, N_REF = 123, 2_000_000
SEED_SINGLE = p.seed
N_REP = 200
REP_SEEDS = range(N_REP)
SCHEDULES = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 4, 8), (0, 1, 2, 4, 8, 16)]

# ---------------- software versions ----------------
try:
    import qiskit
    QISKIT = qiskit.__version__
except ImportError:
    qiskit = None
    QISKIT = None
V["python"] = platform.python_version()
V["numpy"] = np.__version__
V["scipy"] = scipy.__version__
V["qiskit"] = QISKIT

def debt_T_vec(G, R):
    d = np.full_like(G, p.d0, dtype=float)
    g = p.g_bar + G / p.T
    r = p.r_bar + R / p.T
    for _ in range(p.T):
        d = (1 + (r - g) / (1 + g)) * d - (p.a + p.b * (d - p.d_star) + p.pb_policy)
    return d

# ======================================================================
# Table 1: parameters
# ======================================================================
lam0 = (p.r_bar - p.g_bar) / (1 + p.g_bar)
V.update(T=p.T, d0=p.d0, dstar=p.d_star, a=p.a, b=p.b, pbpol=p.pb_policy,
         gbar=p.g_bar, rbar=p.r_bar, sg=p.sigma_g, sr=p.sigma_r,
         sG=pr["sdG"], sR=pr["sdR"], lam0=lam0, pb1=p.a + p.b * (p.d0 - p.d_star),
         dT_nom=debt_T(0, 0, p), n=n, m=p.sum_bits, thr=p.threshold,
         shots=p.shots, powers=list(p.grover_powers), nsig=p.n_sigma)
w("## Table 1 - Model parameters")
w("| Symbol | Meaning | Value |\n|---|---|---|")
for r_ in [
    ("T", "horizon (years)", p.T), ("d_0", "initial debt/GDP", p.d0),
    ("d*", "target ratio of the fiscal rule", p.d_star),
    ("a", "intercept of the primary-balance rule", p.a),
    ("b", "feedback of pb on the debt gap", p.b),
    ("pb_policy", "constant policy effect on pb (%GDP/yr)", p.pb_policy),
    ("g-bar, r-bar", "trend growth, mean interest rate", f"{p.g_bar}, {p.r_bar}"),
    ("sigma_g, sigma_r", "std of the yearly growth / rate shocks", f"{p.sigma_g}, {p.sigma_r}"),
    ("E[G], E[R]", "means of the cumulative shocks", "0, 0"),
    ("sigma_G, sigma_R", "std of G, R (= sigma*sqrt(T))", f"{pr['sdG']:.5f}, {pr['sdR']:.5f}"),
    ("lambda (nominal)", "(r-bar - g-bar)/(1+g-bar)", f"{lam0:.6f}"),
    ("pb_1 (nominal)", "b(d_0 - d*)", f"{V['pb1']:.4f}"),
    ("d_T (nominal)", "terminal debt ratio at G=R=0", f"{V['dT_nom']:.4f}"),
    ("tau", "threshold of the event d_T > tau", p.threshold),
    ("n", "qubits per random factor", n), ("m", "qubits of the sum register", p.sum_bits),
    ("k set", "Grover powers", str(p.grover_powers)), ("shots", "shots per power", p.shots),
]: w(f"| {r_[0]} | {r_[1]} | {r_[2]} |")

# ======================================================================
# Table 2: truncation + linearisation
# ======================================================================
dmin = debt_T(+pr['hi1'], -pr['hi2'], p); dmax = debt_T(-pr['hi1'], +pr['hi2'], p)
G_g, R_g = np.meshgrid(-pr['grid1'], pr['grid2'], indexing="ij")
d_exact_grid = debt_T_vec(G_g, R_g)
I, J = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
S_grid = np.vectorize(lambda i, j: int_sum(int(i), int(j), pr))(I, J).astype(int)
d_lin_grid = pr['d_const'] + pr['c1'] * I + pr['c2'] * J
d_int_grid = pr['d_const'] + pr['delta'] * S_grid
V.update(
    G_lo=-pr['hi1'], G_hi=pr['hi1'], R_lo=pr['lo2'], R_hi=pr['hi2'],
    cut=2 * norm.cdf(-p.n_sigma), stepG=pr['step1'], stepR=pr['step2'],
    dG=-pr['c1_per_unit'], dR=pr['c2_per_unit'], c1=pr['c1'], c2=pr['c2'],
    dconst=pr['d_const'], c1_sigma=pr['c1_per_unit'] * pr['sdG'],
    c2_sigma=pr['c2_per_unit'] * pr['sdR'], dmin_exact=dmin, dmax_exact=dmax,
    dmin_lin=pr['d_const'], dmax_lin=pr['d_const'] + (pr['c1'] + pr['c2']) * (M - 1),
    err_lin_max=float(np.abs(d_lin_grid - d_exact_grid).max()),
    err_int_max=float(np.abs(d_int_grid - d_exact_grid).max()))
w("\n## Table 2 - Truncation and linearisation")
w("| Quantity | Value |\n|---|---|")
for k_, v_ in [
    ("Truncation range of G", f"[{V['G_lo']:.5f}, {V['G_hi']:.5f}] (= +/-3 sigma_G)"),
    ("Truncation range of R", f"[{V['R_lo']:.5f}, {V['R_hi']:.5f}] (= +/-3 sigma_R)"),
    ("Mass cut per variable", f"{V['cut']:.4f}"),
    ("Grid step of G / R", f"{V['stepG']:.5f} / {V['stepR']:.5f}"),
    ("dd_T/dG, dd_T/dR (per unit)", f"{V['dG']:+.4f} (negative) / {V['dR']:+.4f} (positive)"),
    ("same, per standard deviation", f"{-V['c1_sigma']:+.5f} / {V['c2_sigma']:+.5f}"),
    ("c1, c2 (increase of d per grid step of u1=-G, u2=+R)", f"{V['c1']:+.6f}, {V['c2']:+.6f}"),
    ("d_const (intercept at x1=x2=0)", f"{V['dconst']:.6f}"),
    ("exact d_T at best / worst corner", f"{dmin:.4f} / {dmax:.4f}"),
    ("linear model at best / worst corner", f"{V['dmin_lin']:.4f} / {V['dmax_lin']:.4f}"),
    ("max |linear - exact| of d_T on the grid", f"{V['err_lin_max']:.4f}"),
    ("max |integer circuit - exact| of d_T on the grid", f"{V['err_int_max']:.4f}"),
]: w(f"| {k_} | {v_} |")

# ======================================================================
# Table 3: integer arithmetic
# ======================================================================
S_max = sum(pr['w1']) + sum(pr['w2'])
eff_thr = pr['d_const'] + pr['thr_int'] * pr['delta']
V.update(delta=pr['delta'], w1=pr['w1'], w2=pr['w2'], S_max=S_max,
         cap=2 ** p.sum_bits - 1, slack=2 ** p.sum_bits - 1 - S_max,
         thr_int=pr['thr_int'], eff_thr=eff_thr)
w("\n## Table 3 - Integer arithmetic")
w("| Quantity | Value |\n|---|---|")
for k_, v_ in [
    ("delta (debt per integer unit)", f"{pr['delta']:.6f}"),
    ("w1 (per bit j=0..n-1)", pr['w1']), ("w2 (per bit j=0..n-1)", pr['w2']),
    ("max S", S_max), ("capacity 2^m - 1", V['cap']),
    ("overflow check", f"{'OK' if S_max <= V['cap'] else 'OVERFLOW'} (slack {V['slack']})"),
    ("tau_int (flag S >= tau_int)", pr['thr_int']),
    ("effective threshold d_const + tau_int*delta", f"{eff_thr:.4f} (stated tau = {p.threshold})"),
]: w(f"| {k_} | {v_} |")

# ======================================================================
# Table 4: probabilities and errors
# ======================================================================
a_circ, a_grid, a_cont = (float(v) for v in reference_probabilities(p, pr))
se_cont = math.sqrt(a_cont * (1 - a_cont) / N_REF)
mu_lin = debt_T(0, 0, p)
sig_lin = math.hypot(V['c1_sigma'], V['c2_sigma'])
a_lincont = float(1 - norm.cdf((p.threshold - mu_lin) / sig_lin))
rng = np.random.default_rng(SEED_REF)
Gt = truncnorm.rvs(-p.n_sigma, p.n_sigma, scale=pr['sdG'], size=N_REF, random_state=rng)
Rt = truncnorm.rvs(-p.n_sigma, p.n_sigma, scale=pr['sdR'], size=N_REF, random_state=rng)
a_trunc = float((debt_T_vec(Gt, Rt) > p.threshold).mean())

ideal = make_ideal_prob_fn(a_circ)
a_hat, calls = mlae(ideal, p.grover_powers, p.shots, SEED_SINGLE)
est = np.array([mlae(ideal, p.grover_powers, p.shots, seed=s)[0] for s in REP_SEEDS])
bias = float(est.mean() - a_circ); std = float(est.std(ddof=1))
rmse = float(math.sqrt(np.mean((est - a_circ) ** 2)))
V.update(a_cont=a_cont, se_cont=se_cont, a_lincont=a_lincont, mu_lin=mu_lin, sig_lin=sig_lin,
         a_trunc=a_trunc, a_grid=a_grid, a_circ=a_circ, a_hat=float(a_hat),
         E_model=a_circ - a_cont, E_trunc=a_trunc - a_cont, E_disc=a_grid - a_trunc,
         E_lin_int=a_circ - a_grid, E_lin_cont=a_cont - a_lincont,
         E_qae=float(a_hat) - a_circ, E_total=float(a_hat) - a_cont,
         rmse=rmse, bias=bias, std=std, calls=calls, n_rep=N_REP,
         seed_ref=SEED_REF, n_ref=N_REF, seed_single=SEED_SINGLE,
         exec_runs=len(p.grover_powers) * p.shots,
         sum2k1=sum(2 * k + 1 for k in p.grover_powers))
w("\n## Table 4 - a = P(d_T > tau) and its errors")
w("| Quantity | Value |\n|---|---|")
for k_, v_ in [
    (f"a_cont: continuous Gaussians, exact model ({N_REF:,} samples, seed {SEED_REF})", f"{a_cont:.4f} (+/-{se_cont:.4f})"),
    ("a_lin,cont: continuous Gaussians, linear model (closed form)", f"{a_lincont:.4f}"),
    ("a_trunc: truncated at +/-3 sigma, exact model", f"{a_trunc:.4f}"),
    (f"a_grid: {M}x{M} grid, exact model", f"{a_grid:.4f}"),
    ("a_circ: grid + linear + integer (circuit target)", f"{a_circ:.4f}"),
    (f"a_hat: QAE, single run (seed {SEED_SINGLE})", f"{a_hat:.4f}"),
    ("Linearisation, continuous (a_cont - a_lin,cont)", f"{V['E_lin_cont']:+.4f}"),
    ("E_model = a_circ - a_cont", f"{V['E_model']:+.4f}"),
    ("  truncation (a_trunc - a_cont)", f"{V['E_trunc']:+.4f}"),
    (f"  discretisation, {M} points (a_grid - a_trunc)", f"{V['E_disc']:+.4f}"),
    ("  linearisation + integer on the grid (a_circ - a_grid)", f"{V['E_lin_int']:+.4f}"),
    ("E_QAE = a_hat - a_circ (single run)", f"{V['E_qae']:+.4f}"),
    ("E_total = a_hat - a_cont", f"{V['E_total']:+.4f}"),
    (f"RMSE of a_hat vs a_circ ({N_REP} repetitions, seeds 0-{N_REP-1})", f"{rmse:.4f}"),
    ("bias / std of a_hat", f"{bias:+.4f} / {std:.4f}"),
    ("calls to A per run = shots*sum(2k+1)", f"{calls} ({p.shots}*{V['sum2k1']})"),
    ("circuit executions per run", V['exec_runs']),
]: w(f"| {k_} | {v_} |")

# ======================================================================
# Table 5: RMSE vs oracle calls with error bars and slope
# ======================================================================
rows = []
for sch in SCHEDULES:
    e = np.array([mlae(ideal, sch, p.shots, seed=s)[0] for s in REP_SEEDS])
    calls_s = p.shots * sum(2 * k + 1 for k in sch)
    r_q = math.sqrt(np.mean((e - a_circ) ** 2))
    se = r_q / math.sqrt(2 * (N_REP - 1))
    r_c = math.sqrt(a_circ * (1 - a_circ) / calls_s)
    rows.append(dict(sched=str(sch), calls=calls_s, rmse=r_q, se=se, classical=r_c))
x = np.log([r_['calls'] for r_ in rows]); y = np.log([r_['rmse'] for r_ in rows])
A_ = np.vstack([x, np.ones_like(x)]).T
coef, res, *_ = np.linalg.lstsq(A_, y, rcond=None)
slope_se = math.sqrt(float(res[0]) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
V['rmse_table'] = rows; V['slope'] = float(coef[0]); V['slope_se'] = float(slope_se)
w("\n## Table 5 - RMSE vs oracle calls (%d repetitions, %d shots/power)" % (N_REP, p.shots))
w("| Schedule | N_calls | RMSE (MLAE) +/- s.e. | Classical RMSE at N=N_calls |\n|---|---|---|---|")
for r_ in rows:
    w(f"| {r_['sched']} | {r_['calls']:,} | {r_['rmse']:.5f} +/- {r_['se']:.5f} | {r_['classical']:.5f} |")
w(f"\nLog-log slope of MLAE RMSE vs N_calls: {coef[0]:.2f} +/- {slope_se:.2f} (1 s.e.; classical = -0.50)")
eps = 0.005
V['classical_samples_eps'] = a_circ * (1 - a_circ) / eps ** 2
V['qae_bound_eps'] = math.pi / (4 * eps)
w(f"For eps={eps}: classical a(1-a)/eps^2 = {V['classical_samples_eps']:,.0f} samples; "
  f"pi/(4 eps) = {V['qae_bound_eps']:.0f} (asymptotic bound, no constants)")

# ======================================================================
# Table 6: sweep over n
# ======================================================================
w("\n## Table 6 - Sweep over n (m = n + 1, shots = %d)" % p.shots)
w("| n | grid | m | tau_int | effective tau | max S / cap | a_grid | a_circ | a_hat (ideal MLAE, seed 1) | a_circ - a_cont |\n|---|---|---|---|---|---|---|---|---|---|")
sweep = []
for nn in (3, 4, 5):
    if nn == n:
        pp, pp_r, refs = p, pr, (a_circ, a_grid, a_cont)
    else:
        pp = P(n_bits=nn, sum_bits=nn + 1)
        pp_r = build_problem(pp); refs = tuple(float(v) for v in reference_probabilities(pp, pp_r))
    ah = mlae(make_ideal_prob_fn(refs[0]), pp.grover_powers, pp.shots, SEED_SINGLE)[0]
    Sm = sum(pp_r['w1']) + sum(pp_r['w2'])
    effn = pp_r['d_const'] + pp_r['thr_int'] * pp_r['delta']
    sweep.append(dict(n=nn, grid=2 ** nn, m=pp.sum_bits, tau_int=pp_r['thr_int'], eff=effn,
                      Smax=Sm, cap=2 ** pp.sum_bits - 1, a_grid=refs[1], a_circ=refs[0],
                      a_hat=float(ah), err=refs[0] - refs[2]))
    s_ = sweep[-1]
    w(f"| {nn} | {s_['grid']}x{s_['grid']} | {s_['m']} | {s_['tau_int']} | {effn:.4f} | {Sm}/{s_['cap']} | "
      f"{refs[1]:.4f} | {refs[0]:.4f} | {ah:.4f} | {refs[0]-refs[2]:+.4f} |")
V['sweep'] = sweep

# ======================================================================
# Optional Qiskit-only items
# ======================================================================
# V['grover_dev'] = None; V['gates'] = None; V['nqubits'] = None
# if qiskit is not None:
V['grover_dev'] = None; V['gates'] = None; V['nqubits'] = None; V['sim_desc'] = None; V['p0_err'] = None
CACHE_F = "qiskit_cache.json"
KEY = f"n{n}_m{p.sum_bits}_thr{p.threshold}_prec{args.precision}_k{args.kmax}"
cache = json.load(open(CACHE_F)) if os.path.exists(CACHE_F) else {}
if KEY in cache and not args.force_qiskit:
    V.update(cache[KEY]); print(f"[Qiskit results loaded from {CACHE_F}; use --force-qiskit to recompute]")
need_sim = V['p0_err'] is None
need_gates = args.gates and V['gates'] is None
if qiskit is not None and args.qiskit and not args.skip_qiskit and (need_sim or need_gates):
    try:
        from quantum_mc_debt import build_A, grover_probabilities
        t0 = time.time()
        A, obj, total = build_A(pr); V['nqubits'] = total
        if need_sim:
            probs, V['sim_desc'] = grover_probabilities(pr, args.kmax, args.device, args.precision)
            th = math.asin(math.sqrt(probs[0]))
            V['grover_dev'] = (max(abs(probs[k] - math.sin((2 * k + 1) * th) ** 2) for k in probs)
                               if len(probs) > 1 else None)
            V['kmax'] = args.kmax
            V['p0_err'] = abs(probs[0] - a_circ)          # circuit vs classical a_circ
        if need_gates:

            from qiskit import transpile
        # A, obj, total = build_A(pr)
        # pf = make_grover_prob_fn(pr)
        # th = math.asin(math.sqrt(a_circ))
        # V['grover_dev'] = max(abs(pf(k) - math.sin((2 * k + 1) * th) ** 2) for k in range(0, 6))
            basis = ['cx', 'rz', 'sx', 'x']
            tc = transpile(A, basis_gates=basis, optimization_level=1)
            V['gates'] = dict(basis=basis, depth=tc.depth(), cx=tc.count_ops().get('cx', 0),
                          ops=dict(tc.count_ops()), connectivity="all-to-all")
        # V['nqubits'] = total
        cache[KEY] = {k: V.get(k) for k in ("grover_dev", "nqubits", "gates", "sim_desc", "p0_err", "kmax")}
        json.dump(cache, open(CACHE_F, "w"), indent=1, default=float)
        print(f"[Qiskit part took {time.time() - t0:.0f} s]")
    except Exception as ex:
        print("[qiskit-only items failed:", ex, "]")
w("\n## Qiskit-only items")
w(f"- qubits of A: {V['nqubits'] if V['nqubits'] else 'not run (needs Qiskit)'}")
w(f"- max |P_circuit(k) - sin^2((2k+1) theta)|, k=0..5: {V['grover_dev'] if V['grover_dev'] is not None else 'not run (needs Qiskit)'}")
w(f"- |P_circuit(k=0) - a_circ| (circuit vs classical target): {V['p0_err'] if V['p0_err'] is not None else 'not run'}")
w(f"- simulator: {V['sim_desc'] or 'not run'}")
w(f"- transpiled A: {V['gates'] if V['gates'] else 'not run (use --gates)'}")
# w(f"- transpiled A: {V['gates'] if V['gates'] else 'not run (needs Qiskit)'}")

w("\n## Table 7 - Software and reproducibility")
w(f"- Python {V['python']}, NumPy {V['numpy']}, SciPy {V['scipy']}, Qiskit {QISKIT or '(print qiskit.__version__)'}; Aer not used")
w("- RNG: numpy.random.default_rng (PCG64); reference MC seed %d (%s samples); single QAE run seed %d; %d repetitions seeds 0..%d"
  % (SEED_REF, f"{N_REF:,}", SEED_SINGLE, N_REP, N_REP - 1))
w("- State loading: qiskit.circuit.library.StatePreparation on sqrt(p_i); arithmetic: WeightedAdder + IntegerComparator")
w("- Simulation: exact Statevector probabilities; shots drawn as numpy binomial samples")

open("paper_tables.md", "w").write("\n".join(out))
json.dump(V, open("paper_values.json", "w"), indent=1, default=float)
