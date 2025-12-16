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

 Black–Scholes smoothing is **intentionally excluded** for American puts due to the absence of a closed-form benchmark and excessive computational cost.

---

## 📊 Benchmarks & Validation

| Instrument | Benchmark |
|----------|-----------|
| European Call / Put | Black–Scholes–Merton analytical solution |
| American Call | Black–Scholes (no dividends ⇒ no early exercise) |
| American Put | High-step CRR lattice (N = 10⁴) |

This ensures **theoretical consistency** and **numerical reliability**.

---
### Best Performing Combinations

**European Options & American Calls**
- Leisen–Reimer + Richardson Extrapolation
- CRR + Averaging Smoothing + Richardson
- CRR Pegging Strike + BS Smoothing + Richardson

**American Puts**
- Leisen–Reimer + Richardson
- CRR + Averaging Smoothing + Richardson
- CRR Pegging Strike + Averaging + Richardson

### Error Orders Observed (ALL European + American Call)
- CRR (raw):         O(1/N)
- CRR + Averaging:    O(1/N²)
- Leisen–Reimer:     Near-quadratic
- Richardson Extrapolated: Order boosted by +1

---

## Stress Testing & Robustness

A secondary experiment uses:
- **Higher volatility (σ = 40%)**
- **Non-ATM strike**
- **Shorter maturity**

Results show:
- Grid spacing significantly impacts accuracy
- Pegging-the-strike may degrade under high volatility
- Richardson extrapolation fails if error is not smooth

This highlights **real-world numerical fragility**, not just theoretical behavior.


## Code Structure

- Separate implementations for **CRR**, **Tian**, and **Leisen–Reimer**
- Modular functions for smoothing and extrapolation
- Explicit backward induction (no black-box solvers)
- Designed for **comparability and numerical analysis**

Detailed implementation choices and convergence plots are discussed in the accompanying report.

---

## How to Run
1. Make sure you have Python 3 installed.
2. Install the required libraries (if not already installed):
```bash
pip install numpy scipy matplotlib pandas
