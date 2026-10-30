## Table 1 - Model parameters
| Symbol | Meaning | Value |
|---|---|---|
| T | horizon (years) | 20 |
| d_0 | initial debt/GDP | 0.9 |
| d* | target ratio of the fiscal rule | 0.6 |
| a | intercept of the primary-balance rule | 0.0 |
| b | feedback of pb on the debt gap | 0.03 |
| pb_policy | constant policy effect on pb (%GDP/yr) | 0.0 |
| g-bar, r-bar | trend growth, mean interest rate | 0.015, 0.025 |
| sigma_g, sigma_r | std of the yearly growth / rate shocks | 0.01, 0.005 |
| E[G], E[R] | means of the cumulative shocks | 0, 0 |
| sigma_G, sigma_R | std of G, R (= sigma*sqrt(T)) | 0.04472, 0.02236 |
| lambda (nominal) | (r-bar - g-bar)/(1+g-bar) | 0.009852 |
| pb_1 (nominal) | b(d_0 - d*) | 0.0090 |
| d_T (nominal) | terminal debt ratio at G=R=0 | 0.8978 |
| tau | threshold of the event d_T > tau | 0.95 |
| n | qubits per random factor | 5 |
| m | qubits of the sum register | 6 |
| k set | Grover powers | (0, 1, 2, 4) |
| shots | shots per power | 500 |

## Table 2 - Truncation and linearisation
| Quantity | Value |
|---|---|
| Truncation range of G | [-0.13416, 0.13416] (= +/-3 sigma_G) |
| Truncation range of R | [-0.06708, 0.06708] (= +/-3 sigma_R) |
| Mass cut per variable | 0.0027 |
| Grid step of G / R | 0.00866 / 0.00433 |
| dd_T/dG, dd_T/dR (per unit) | -0.7422 (negative) / +0.7349 (positive) |
| same, per standard deviation | -0.03319 / +0.01643 |
| c1, c2 (increase of d per grid step of u1=-G, u2=+R) | +0.006424, +0.003180 |
| d_const (intercept at x1=x2=0) | 0.748922 |
| exact d_T at best / worst corner | 0.7624 / 1.0620 |
| linear model at best / worst corner | 0.7489 / 1.0467 |
| max |linear - exact| of d_T on the grid | 0.0154 |
| max |integer circuit - exact| of d_T on the grid | 0.0163 |

## Table 3 - Integer arithmetic
| Quantity | Value |
|---|---|
| delta (debt per integer unit) | 0.004726 |
| w1 (per bit j=0..n-1) | [1, 3, 5, 11, 22] |
| w2 (per bit j=0..n-1) | [1, 1, 3, 5, 11] |
| max S | 63 |
| capacity 2^m - 1 | 63 |
| overflow check | OK (slack 0) |
| tau_int (flag S >= tau_int) | 43 |
| effective threshold d_const + tau_int*delta | 0.9521 (stated tau = 0.95) |

## Table 4 - a = P(d_T > tau) and its errors
| Quantity | Value |
|---|---|
| a_cont: continuous Gaussians, exact model (2,000,000 samples, seed 123) | 0.0861 (+/-0.0002) |
| a_lin,cont: continuous Gaussians, linear model (closed form) | 0.0793 |
| a_trunc: truncated at +/-3 sigma, exact model | 0.0845 |
| a_grid: 32x32 grid, exact model | 0.0818 |
| a_circ: grid + linear + integer (circuit target) | 0.0818 |
| a_hat: QAE, single run (seed 1) | 0.0809 |
| Linearisation, continuous (a_cont - a_lin,cont) | +0.0067 |
| E_model = a_circ - a_cont | -0.0043 |
|   truncation (a_trunc - a_cont) | -0.0015 |
|   discretisation, 32 points (a_grid - a_trunc) | -0.0027 |
|   linearisation + integer on the grid (a_circ - a_grid) | +0.0000 |
| E_QAE = a_hat - a_circ (single run) | -0.0009 |
| E_total = a_hat - a_cont | -0.0052 |
| RMSE of a_hat vs a_circ (200 repetitions, seeds 0-199) | 0.0010 |
| bias / std of a_hat | -0.0000 / 0.0011 |
| calls to A per run = shots*sum(2k+1) | 9000 (500*18) |
| circuit executions per run | 2000 |

## Table 5 - RMSE vs oracle calls (200 repetitions, 500 shots/power)
| Schedule | N_calls | RMSE (MLAE) +/- s.e. | Classical RMSE at N=N_calls |
|---|---|---|---|
| (0,) | 500 | 0.01286 +/- 0.00064 | 0.01226 |
| (0, 1) | 2,000 | 0.00410 +/- 0.00021 | 0.00613 |
| (0, 1, 2) | 4,500 | 0.00205 +/- 0.00010 | 0.00409 |
| (0, 1, 2, 4) | 9,000 | 0.00105 +/- 0.00005 | 0.00289 |
| (0, 1, 2, 4, 8) | 17,500 | 0.00071 +/- 0.00004 | 0.00207 |
| (0, 1, 2, 4, 8, 16) | 34,000 | 0.00033 +/- 0.00002 | 0.00149 |

Log-log slope of MLAE RMSE vs N_calls: -0.85 +/- 0.02 (1 s.e.; classical = -0.50)
For eps=0.005: classical a(1-a)/eps^2 = 3,004 samples; pi/(4 eps) = 157 (asymptotic bound, no constants)

## Table 6 - Sweep over n (m = n + 1, shots = 500)
| n | grid | m | tau_int | effective tau | max S / cap | a_grid | a_circ | a_hat (ideal MLAE, seed 1) | a_circ - a_cont |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 8x8 | 4 | 11 | 0.9673 | 15/15 | 0.0611 | 0.0611 | 0.0617 | -0.0250 |
| 4 | 16x16 | 5 | 21 | 0.9506 | 31/31 | 0.0751 | 0.0834 | 0.0825 | -0.0027 |
| 5 | 32x32 | 6 | 43 | 0.9521 | 63/63 | 0.0818 | 0.0818 | 0.0809 | -0.0043 |

## Qiskit-only items
- qubits of A: not run (needs Qiskit)
- max |P_circuit(k) - sin^2((2k+1) theta)|, k=0..5: not run (needs Qiskit)
- |P_circuit(k=0) - a_circ| (circuit vs classical target): not run
- simulator: not run
- transpiled A: not run (use --gates)

## Table 7 - Software and reproducibility
- Python 3.14.7, NumPy 2.5.4, SciPy 1.18.1, Qiskit 2.5.2; Aer not used
- RNG: numpy.random.default_rng (PCG64); reference MC seed 123 (2,000,000 samples); single QAE run seed 1; 200 repetitions seeds 0..199
- State loading: qiskit.circuit.library.StatePreparation on sqrt(p_i); arithmetic: WeightedAdder + IntegerComparator
- Simulation: exact Statevector probabilities; shots drawn as numpy binomial samples