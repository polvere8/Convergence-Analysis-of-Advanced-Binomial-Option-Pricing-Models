# Convergence-Analysis-of-Advanced-Binomial-Option-Pricing-Models

This project explores **binomial tree models for option pricing**, focusing on **convergence, accuracy, and numerical stability** for both **European and American options** (calls and puts).

We implement and compare three classical binomial models:

- **Cox–Ross–Rubinstein (CRR)**
- **Tian**
- **Leisen–Reimer (LR)**

and enhance them using several **smoothing techniques** and **numerical acceleration methods**, including **Richardson extrapolation**, to mitigate oscillations (sawtooth effect) and improve convergence.


## Numerical Enhancements

The following techniques are implemented **individually and in combination**:

| Technique | Purpose |
|---------|--------|
| Black–Scholes smoothing | Smooth terminal payoff discontinuity |
| Averaging smoothing | Cancel odd–even oscillations |
| Pegging the strike | Align strike with terminal node |
| Richardson extrapolation | Increase order of convergence |

For American options, Black–Scholes smoothing uses on the penultimate layer the maximum between the European Black–Scholes price over one step and the exercise value.

---

## 📊 Benchmarks & Validation

| Instrument | Benchmark |
|----------|-----------|
| European Call / Put | Black–Scholes–Merton analytical solution |
| American Call | Black–Scholes (no dividends ⇒ no early exercise) |
| American Put | High-step Leisen–Reimer + Richardson lattice (N = 5001), cross-checked with CRR + BS smoothing + Richardson (N = 5000) |

This ensures **theoretical consistency** and **numerical reliability**.

---
### Best Performing Combinations

**European Options & American Calls**
- Leisen–Reimer + Richardson Extrapolation
- CRR + Averaging Smoothing + Richardson (base case)
- CRR Pegging Strike + Averaging + Richardson

**American Puts**
- Leisen–Reimer + Richardson
- CRR + Averaging Smoothing + Richardson (base case)
- CRR Pegging Strike + Averaging + Richardson

### Error Orders Observed (ALL European + American Call)
- CRR (raw):         O(1/N)
- CRR + Averaging:    O(1/N), oscillations removed
- Leisen–Reimer:     Quadratic, O(1/N²)
- Richardson Extrapolated: Order boosted by +1 only if the error is smooth (LR + RE: O(1/N³), CRR + Averaging + RE: O(1/N²), raw CRR + RE: no gain)

For American puts, the early-exercise boundary reduces Leisen–Reimer to O(1/N), so Richardson uses 2·P(2N+1) − P(N).

---

## Stress Testing & Robustness

A secondary experiment uses:
- **Higher volatility (σ = 40%)**
- **Non-ATM strike**
- **Shorter maturity**

Results show:
- Grid spacing significantly impacts accuracy
- Pegging the strike keeps the error smooth, so pegging-based combinations remain accurate
- Richardson extrapolation fails if error is not smooth (CRR + Averaging + Richardson loses almost two orders of magnitude)

This highlights **real-world numerical fragility**, not just theoretical behavior.


## Code Structure

- Separate implementations for **CRR**, **Tian**, and **Leisen–Reimer**
- Modular functions for smoothing and extrapolation
- Explicit backward induction (no black-box solvers), vectorised with NumPy
- Designed for **comparability and numerical analysis**
- Set `CASE = "base"` or `CASE = "seminar"` at the top of each file to switch parameters

Detailed implementation choices and convergence plots are discussed in the accompanying report.

---

## How to Run
1. Make sure you have Python 3 installed.
2. Install the required libraries (if not already installed):
```bash
pip install numpy scipy matplotlib
```
3. Run any of the scripts, for example:
```bash
python EuropeanCALL.py
```
