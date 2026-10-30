"""Fills paper_template.tex with the numbers of paper_values.json -> paper_revised.tex
Run order:  python paper_tables.py   then   python build_paper.py"""
import json, math, re

V = json.load(open("paper_values.json"))
F = {}

def f4(x): return f"{x:.4f}"
def f5(x): return f"{x:.5f}"
def sg(x, k=4):                      # signed number usable in text or math mode
    s = f"{abs(x):.{k}f}"
    return "\\ensuremath{" + ("-" if x < 0 else "+") + s + "}"
def thou(n): return f"{int(n):,}".replace(",", "\\,")

# ---- direct values ----
for k in ["T", "d0", "dstar", "a", "b", "pbpol", "gbar", "rbar", "sg", "sr", "n", "m", "shots", "nsig",
          "calls", "exec_runs", "sum2k1", "n_rep", "seed_ref", "seed_single", "S_max", "cap", "slack", "thr_int"]:
    F[k] = str(V[k])
F["thr"] = f"{V['thr']:.2f}"
F["n_ref"] = thou(V["n_ref"]); F["calls"] = thou(V["calls"]); F["exec_runs"] = thou(V["exec_runs"])
for k, d in [("sG", 5), ("sR", 5), ("lam0", 6), ("pb1", 4), ("dT_nom", 4), ("cut", 4), ("stepG", 5), ("stepR", 5),
             ("dconst", 6), ("dmin_exact", 4), ("dmax_exact", 4), ("dmin_lin", 4), ("dmax_lin", 4),
             ("err_lin_max", 4), ("err_int_max", 4), ("delta", 6), ("eff_thr", 4), ("mu_lin", 4), ("sig_lin", 5),
             ("a_cont", 4), ("se_cont", 4), ("a_lincont", 4), ("a_trunc", 4), ("a_grid", 4), ("a_circ", 4), ("a_hat", 4)]:
    F[k] = f"{V[k]:.{d}f}"
for k in ["G_lo", "G_hi", "R_lo", "R_hi"]: F[k] = sg(V[k], 5)
for k in ["dG", "dR"]: F[k] = sg(V[k], 4)
F["dG_sig"] = sg(-V["c1_sigma"], 5); F["dR_sig"] = sg(V["c2_sigma"], 5)
F["c1"] = sg(V["c1"], 6); F["c2"] = sg(V["c2"], 6)
for k in ["E_lin_cont", "E_model", "E_trunc", "E_disc", "E_lin_int", "E_qae", "E_total"]:
    F[k] = sg(V[k], 4)
F["E_lin_cont_abs"] = f4(abs(V["E_lin_cont"]))
F["rmse"] = f5(V["rmse"]); F["bias"] = sg(V["bias"], 5); F["std"] = f5(V["std"])
F["w1"] = str(V["w1"]).replace(" ", ""); F["w2"] = str(V["w2"]).replace(" ", "")
F["powers"] = "$\\{" + ",".join(map(str, V["powers"])) + "\\}$"
F["grid"] = str(2 ** V["n"]); F["gridm1"] = str(2 ** V["n"] - 1); F["grid2"] = thou(4 ** V["n"])
F["n_rep_m1"] = str(V["n_rep"] - 1)
F["slope"] = sg(V["slope"], 2); F["slope_se"] = f"{V['slope_se']:.2f}"
rt = V["rmse_table"]
F["calls_min"] = thou(rt[0]["calls"]); F["calls_max"] = thou(rt[-1]["calls"])
F["ratio_err"] = f"{abs(V['E_model']) / V['rmse']:.1f}"
for s_ in V["sweep"]:
    F[f"eff_n{s_['n']}"] = f4(s_["eff"])
F["sweep_rows"] = "\n".join(
    f"{s_['n']} & ${s_['grid']}\\times{s_['grid']}$ & {s_['m']} & {s_['tau_int']} & {s_['eff']:.4f} & "
    f"{s_['Smax']}/{s_['cap']} & {s_['a_circ']:.4f} & {sg(s_['err'])} \\\\" for s_ in V["sweep"])
F["rmse_rows"] = "\n".join(
    f"$({r['sched'].strip('()').rstrip(',').replace(' ', '')})$ & {thou(r['calls'])} & "
    f"{r['rmse']:.5f} ($\\pm${r['se']:.5f}) & {r['classical']:.5f} \\\\" for r in rt)
main = next(r for r in rt if r["calls"] == V["calls"])
F["rmse_first"] = f5(rt[0]["rmse"]); F["cl_first"] = f5(rt[0]["classical"])
F["rmse_main"] = f5(main["rmse"]); F["cl_main"] = f5(main["classical"])
F["rmse_last"] = f5(rt[-1]["rmse"]); F["cl_last"] = f5(rt[-1]["classical"])
F["ratio_last"] = f"{rt[-1]['classical'] / rt[-1]['rmse']:.1f}"
F["decades"] = f"{math.log10(rt[-1]['calls'] / rt[0]['calls']):.1f}"
F["cl_eps"] = thou(round(V["classical_samples_eps"])); F["qae_eps"] = str(round(V["qae_bound_eps"]))
r01 = next(r for r in rt if r["calls"] == 4 * V["shots"])
F["n2000"] = thou(r01["calls"]); F["rmse_01"] = f5(r01["rmse"])
F["gap_pp"] = f"{(V['thr'] - V['dT_nom']) * 100:.1f}"

# ---- Qiskit-only items (filled only if paper_tables.py ran with Qiskit) ----
g = V.get("gates")
F["A_cost_phrase"] = (f"transpiled depth {g['depth']} and {g['cx']} CNOT gates" if g else
                      "\\todo{transpiled depth and CNOT count of $\\mathcal A$}")
F["gates_paragraph"] = (
    f"Transpiled to the basis gate set $\\{{\\mathrm{{cx,rz,sx,x}}\\}}$ with all-to-all connectivity (optimisation level 1), "
    f"$\\mathcal A$ has depth {g['depth']} and {g['cx']} CNOT gates." if g else
    "\\todo{Report the transpiled depth and CNOT count of $\\mathcal A$ for the stated basis gates and connectivity "
    "(computed by paper\\_tables.py when Qiskit is installed); otherwise omit gate-count claims.}")
F["qubits_sentence"] = (f"The circuit $\\mathcal A$ uses {V['nqubits']} qubits." if V.get("nqubits") else
                        "\\todo{Report the number of qubits of $\\mathcal A$ (paper\\_tables.py, Qiskit).}")
F["grover_sentence"] = (f"The maximum deviation from $\\sin^2((2k+1)\\theta)$ for $k=0,\\dots,5$ is {V['grover_dev']:.1e}." if V.get("grover_dev") is not None else
                        "\\todo{Report the maximum deviation from $\\sin^2((2k+1)\\theta)$ (paper\\_tables.py, Qiskit).}")
F["qiskit_ver"] = V["qiskit"] or "2.5.2"
F["python"] = V["python"]; F["numpy"] = V["numpy"]; F["scipy"] = V["scipy"]

tex = open("paper_template.tex").read()
tex = re.sub(r"<<(\w+)>>", lambda m: F[m.group(1)], tex)
assert "<<" not in tex
open("paper_revised.tex", "w").write(tex)
print("paper_revised.tex written;", len(F), "values filled")
