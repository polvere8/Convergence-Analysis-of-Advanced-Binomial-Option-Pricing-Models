"""
European call option: convergence of binomial tree models to Black-Scholes.

Models:        CRR (Cox-Ross-Rubinstein), Tian, LR (Leisen-Reimer)
Smoothing:     BSS (Black-Scholes smoothing), AS (averaging), PEG (pegging the strike)
Extrapolation: RE (Richardson extrapolation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Parameters: set CASE to "base" or "seminar"
# --------------------------------------------------------------------------- #
CASE = "base"

PARAMS = {
    "base":    dict(S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0),
    "seminar": dict(S0=100.0, K=110.0, r=0.02, sigma=0.40, T=0.5),
}


# --------------------------------------------------------------------------- #
# Black-Scholes formula for a European call (also works with an array S0)
# --------------------------------------------------------------------------- #
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# --------------------------------------------------------------------------- #
# Shared tree helpers
# --------------------------------------------------------------------------- #
def stock_layer(S0, u, d, n):
    """Stock prices after n steps, from the lowest (j = 0) to the highest (j = n)."""
    j = np.arange(n + 1)
    return S0 * u**j * d**(n - j)


def backward_induction(C, p, discount):
    """Roll the option values C back to t = 0.

    At each step, node j of the previous layer gets
        discount * (p * C[j+1] + (1 - p) * C[j]),
    i.e. C[1:] are the up-moves and C[:-1] the down-moves.
    Each step shortens the array by one; at the end only C[0] is left.
    """
    while len(C) > 1:
        C = discount * (p * C[1:] + (1 - p) * C[:-1])
    return C[0]


def price_on_tree(S0, K, r, T, N, u, d, p):
    """European call on a binomial tree with given u, d, p."""
    C = np.maximum(stock_layer(S0, u, d, N) - K, 0.0)   # payoff at maturity
    return backward_induction(C, p, np.exp(-r * T / N))


def price_on_tree_BSS(S0, K, r, sigma, T, N, u, d, p):
    """Black-Scholes smoothing: on the penultimate layer the option value is the
    Black-Scholes price with time to maturity dt (removes the payoff kink)."""
    dt = T / N
    C = black_scholes_call(stock_layer(S0, u, d, N - 1), K, r, sigma, dt)
    return backward_induction(C, p, np.exp(-r * dt))


# --------------------------------------------------------------------------- #
# Lattice parameters
# --------------------------------------------------------------------------- #
def crr_params(r, sigma, T, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    return u, d, p


def peg_params(S0, K, r, sigma, T, N):
    """CRR tree rescaled by gamma so that the final node closest to K lands on K."""
    dt = T / N
    u_crr, d_crr, _ = crr_params(r, sigma, T, N)
    ST = stock_layer(S0, u_crr, d_crr, N)
    S_closest = ST[np.argmin(np.abs(ST - K))]
    gamma = (K / S_closest) ** (1 / N)
    u, d = u_crr * gamma, d_crr * gamma
    p = (np.exp(r * dt) - d) / (u - d)
    return u, d, p


def peizer_pratt(z, N):
    """Peizer-Pratt inversion (method 2), used by Leisen-Reimer."""
    return 0.5 + np.sign(z) * np.sqrt(
        0.25 - 0.25 * np.exp(-(z / (N + 1 / 3))**2 * (N + 1 / 6)))


# --------------------------------------------------------------------------- #
# Binomial models
# --------------------------------------------------------------------------- #
def crr(S0, K, r, sigma, T, N):
    u, d, p = crr_params(r, sigma, T, N)
    return price_on_tree(S0, K, r, T, N, u, d, p)


def tian(S0, K, r, sigma, T, N):
    dt = T / N
    M = np.exp(r * dt)
    V = np.exp(sigma**2 * dt)
    root = np.sqrt(V**2 + 2 * V - 3)
    u = 0.5 * M * V * (V + 1 + root)
    d = 0.5 * M * V * (V + 1 - root)
    p = (M - d) / (u - d)
    return price_on_tree(S0, K, r, T, N, u, d, p)


def LR(S0, K, r, sigma, T, N):
    if N % 2 == 0:
        raise ValueError("Leisen-Reimer needs an odd number of steps")
    dt = T / N
    a = np.exp(r * dt)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p = peizer_pratt(d2, N)
    P = peizer_pratt(d1, N)
    u = a * P / p
    d = a * (1 - P) / (1 - p)
    return price_on_tree(S0, K, r, T, N, u, d, p)


# --------------------------------------------------------------------------- #
# Smoothing techniques and Richardson extrapolation
# --------------------------------------------------------------------------- #
def crr_BSS(S0, K, r, sigma, T, N):          # Black-Scholes smoothing
    u, d, p = crr_params(r, sigma, T, N)
    return price_on_tree_BSS(S0, K, r, sigma, T, N, u, d, p)


def crr_AS(S0, K, r, sigma, T, N):           # Averaging smoothing
    return 0.5 * (crr(S0, K, r, sigma, T, N) + crr(S0, K, r, sigma, T, N + 1))


def crr_RE(S0, K, r, sigma, T, N):           # Richardson extrapolation, error O(1/N)
    return 2 * crr(S0, K, r, sigma, T, 2 * N) - crr(S0, K, r, sigma, T, N)


def LR_RE(S0, K, r, sigma, T, N):            # Richardson for LR, error O(1/N^2)
    # 2N+1 keeps the number of steps odd
    return (4 * LR(S0, K, r, sigma, T, 2 * N + 1) - LR(S0, K, r, sigma, T, N)) / 3


def crr_PEG(S0, K, r, sigma, T, N):          # Pegging the strike
    u, d, p = peg_params(S0, K, r, sigma, T, N)
    return price_on_tree(S0, K, r, T, N, u, d, p)


def crr_BSS_RE(S0, K, r, sigma, T, N):
    return 2 * crr_BSS(S0, K, r, sigma, T, 2 * N) - crr_BSS(S0, K, r, sigma, T, N)


def crr_AS_RE(S0, K, r, sigma, T, N):
    return 2 * crr_AS(S0, K, r, sigma, T, 2 * N) - crr_AS(S0, K, r, sigma, T, N)


def crr_PEG_RE(S0, K, r, sigma, T, N):
    return 2 * crr_PEG(S0, K, r, sigma, T, 2 * N) - crr_PEG(S0, K, r, sigma, T, N)


def crr_PEG_BSS(S0, K, r, sigma, T, N):
    u, d, p = peg_params(S0, K, r, sigma, T, N)
    return price_on_tree_BSS(S0, K, r, sigma, T, N, u, d, p)


def crr_PEG_BSS_RE(S0, K, r, sigma, T, N):
    return 2 * crr_PEG_BSS(S0, K, r, sigma, T, 2 * N) - crr_PEG_BSS(S0, K, r, sigma, T, N)


def crr_PEG_AS(S0, K, r, sigma, T, N):
    return 0.5 * (crr_PEG(S0, K, r, sigma, T, N) + crr_PEG(S0, K, r, sigma, T, N + 1))


def PEG_AS_RE(S0, K, r, sigma, T, N):
    return 2 * crr_PEG_AS(S0, K, r, sigma, T, 2 * N) - crr_PEG_AS(S0, K, r, sigma, T, N)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def new_figure(title, ylabel):
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.xlabel("Number of Steps (N)")
    plt.ylabel(ylabel)
    plt.grid()


def finish_figure(BSM_price=None):
    if BSM_price is not None:
        plt.axhline(BSM_price, linestyle="--", color="black", label="Black–Scholes", alpha=0.7)
    plt.legend()
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Main comparison script
# --------------------------------------------------------------------------- #
def compare_models():
    prm = PARAMS[CASE]

    steps = np.arange(200, 400, 3)
    odd_steps = steps[steps % 2 == 1]       # LR only works with odd N

    BSM_price = black_scholes_call(**prm)

    # Prices for every N
    def prices(method, Ns):
        return np.array([method(N=N, **prm) for N in Ns])

    crr_p            = prices(crr, steps)
    tian_p           = prices(tian, steps)
    LR_p             = prices(LR, odd_steps)
    crr_BSS_p        = prices(crr_BSS, steps)
    crr_AS_p         = prices(crr_AS, steps)
    crr_RE_p         = prices(crr_RE, steps)
    crr_BSS_RE_p     = prices(crr_BSS_RE, steps)
    crr_AS_RE_p      = prices(crr_AS_RE, steps)
    crr_PEG_p        = prices(crr_PEG, steps)
    crr_PEG_RE_p     = prices(crr_PEG_RE, steps)
    crr_PEG_BSS_p    = prices(crr_PEG_BSS, steps)
    crr_PEG_BSS_RE_p = prices(crr_PEG_BSS_RE, steps)
    crr_PEG_AS_p     = prices(crr_PEG_AS, steps)
    PEG_AS_RE_p      = prices(PEG_AS_RE, steps)
    LR_RE_p          = prices(LR_RE, odd_steps)

    # Absolute errors
    crr_e  = np.abs(crr_p - BSM_price)
    tian_e = np.abs(tian_p - BSM_price)
    LR_e   = np.abs(LR_p - BSM_price)

    print(f"Case: {CASE}  {prm}")
    print(f"Black–Scholes price: {BSM_price:.6f}")

    # Price convergence
    # ----------------------------------------- #
    new_figure("CRR Convergence to Black–Scholes", "Call Price")
    plt.plot(steps, crr_p, label="CRR price")
    finish_figure(BSM_price)

    new_figure("Tian Convergence to Black–Scholes", "Call Price")
    plt.plot(steps, tian_p, label="Tian price")
    finish_figure(BSM_price)

    new_figure("LR method Convergence to Black–Scholes", "Call Price")
    plt.plot(odd_steps, LR_p, label="LR price")
    finish_figure(BSM_price)

    # Errors
    # ----------------------------------------- #
    new_figure("Tian error European Call", "Error")
    plt.plot(steps, tian_e, label="Tian error")
    finish_figure()

    new_figure("CRR error European Call", "Error")
    plt.plot(steps, crr_e, label="CRR error")
    plt.plot(steps, 1 / steps, label="1/N")
    finish_figure()

    new_figure("LR error European Call", "Error")
    plt.plot(odd_steps, LR_e, label="LR error")
    finish_figure()

    # Price comparison of the three models
    new_figure("CRR, Tian and LR Convergence to Black–Scholes", "Call Price")
    plt.plot(steps, tian_p, label="Tian price")
    plt.plot(odd_steps, LR_p, label="LR price")
    plt.plot(steps, crr_p, label="CRR price")
    finish_figure(BSM_price)

    # Errors of the three models
    new_figure("Absolute error for the three models, European Call", "Error")
    plt.plot(steps, crr_e, label="CRR")
    plt.plot(steps, tian_e, label="Tian")
    plt.plot(odd_steps, LR_e, label="LR")
    finish_figure()

    # Smoothing
    # -------------------------------------------- #
    new_figure("Black-Scholes Smoothing", "Call Price")
    plt.plot(steps, crr_p, label="CRR price")
    plt.plot(steps, crr_BSS_p, label="CRR + BSS")
    finish_figure(BSM_price)

    new_figure("Averaging Smoothing", "Call Price")
    plt.plot(steps, crr_p, label="CRR price")
    plt.plot(steps, crr_AS_p, label="CRR + AS")
    plt.plot(steps, crr_PEG_AS_p, label="PEG + AS")
    finish_figure(BSM_price)

    new_figure("Pegging the strike", "Call Price")
    plt.plot(steps, crr_PEG_p, label="CRR + PEG")
    plt.plot(steps, crr_p, label="CRR price")
    finish_figure(BSM_price)

    # Richardson extrapolation
    new_figure("Richardson Extrapolation", "Call Price")
    plt.plot(odd_steps, LR_p, label="LR")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    finish_figure(BSM_price)

    # Combination of techniques
    # -------------------------------------------- #
    new_figure("Combination of Techniques Convergence to Black–Scholes", "Call Price")
    plt.plot(odd_steps, LR_p, label="LR")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_p, label="CRR")
    plt.plot(steps, crr_BSS_p, label="CRR + BSS")
    plt.plot(steps, crr_AS_RE_p, label="CRR + AS + RE")
    plt.plot(steps, crr_PEG_p, label="PEG")
    plt.plot(steps, crr_PEG_BSS_p, label="PEG + BSS")
    plt.plot(steps, crr_PEG_BSS_RE_p, label="PEG + BSS + RE")
    plt.plot(steps, PEG_AS_RE_p, label="PEG + AS + RE")
    finish_figure(BSM_price)

    new_figure("BEST METHODS Convergence to Black–Scholes", "Call Price")
    plt.plot(odd_steps, LR_p, label="LR")
    plt.plot(odd_steps, LR_RE_p, label="LR + RE")
    plt.plot(steps, crr_AS_RE_p, label="CRR + AS + RE")
    plt.plot(steps, crr_PEG_BSS_RE_p, label="PEG + BSS + RE")
    plt.plot(steps, PEG_AS_RE_p, label="PEG + AS + RE")
    finish_figure(BSM_price)

    plt.show()


if __name__ == "__main__":
    compare_models()
