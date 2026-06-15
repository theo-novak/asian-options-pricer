# Asian Options Pricer — Monte Carlo & the Lévy Lognormal Approximation

Valuation, Greeks, variance reduction and delta-hedging of **Asian (average-price) options** using
the **Lévy lognormal approximation**, a **modified Black–Scholes–Merton closed form** for geometric
averages, and a **Monte Carlo engine** for arithmetic averages — implemented in NumPy, JAX and
TensorFlow.

> Course project for **FE 620: Pricing & Hedging**, Stevens Institute of Technology.
> Advisor: Dan Pirjol. Authors: Theo Dimitrasopoulos, William Kraemer, Snehal Rajguru, Vaikunth Seshadri.

---

## 1. What the program does

An **Asian option** is an exotic, path-dependent derivative whose payoff depends on the *average*
price of the underlying over the life of the contract, rather than on the spot price at expiry. For a
fixed-strike Asian option with average price \(\bar{A}_T\) and strike \(K\):

```
Call payoff:  C(K, T) = max(Ā_T − K, 0)
Put  payoff:  P(K, T) = max(K − Ā_T, 0)
```

The average can be **arithmetic** \(A_N = \frac{1}{N}\sum_i S_{t_i}\) or **geometric**
\(G_N = \left(\prod_i S_{t_i}\right)^{1/N}\). Because the averaging mechanism dampens volatility, Asian
options are cheaper than the equivalent vanilla European options and are widely used to hedge
exposures in commodities, FX and energy markets (e.g. an airline hedging weekly crude-oil purchases).

This project provides a complete toolkit to:

- **Generate** Geometric Brownian Motion (GBM) price paths.
- **Price** Asian calls and puts with:
  - a **closed-form geometric** (modified Black–Scholes) solution,
  - the **Lévy lognormal approximation** for the arithmetic average,
  - a **Monte Carlo** engine for both arithmetic and geometric averaging.
- **Reduce variance** in Monte Carlo estimates via the geometric payoff as a **control variate**.
- **Compute the Greeks** (1st, 2nd and 3rd order) analytically and via automatic differentiation.
- **Benchmark** accuracy against the **Linetsky (2002)** eigenfunction-expansion test cases.
- **Stress-test** convergence (price error vs. number of Monte Carlo trials).
- Run a **delta-hedging experiment** on simulated intraday data.

---

## 2. The research, in brief

The core valuation difficulty is that **the arithmetic average of lognormal variables is not
lognormal**, so there is no exact closed-form price for arithmetic Asian options and numerical methods
(Monte Carlo) are required. By contrast, the **geometric average of lognormals is lognormal**, so a
geometric Asian option *can* be solved in closed form by adapting the Black–Scholes model
(Kemna & Vorst).

The **Lévy lognormal approximation** sidesteps the non-lognormality of the arithmetic average by
*assuming* the arithmetic average is approximately lognormal and **matching its first two moments**
(mean \(M_1\) and variance \(M_2\) of \(\int_0^T S_t\,dt\) under the risk-neutral measure). This yields
a fast, analytic — but only approximate — price for the arithmetic Asian option.

The notebooks compare these approaches on three axes — **accuracy, robustness and speed** — and reach
the following conclusions (see the report in `report/`):

- The **Lévy approximation** is fast but imprecise.
- A **Monte Carlo Black–Scholes** simulation with **control-variate variance reduction** is highly
  accurate at the cost of higher compute time, matching the Linetsky benchmarks to within
  **0.0147%–0.7345%** absolute error.
- An optimal, time-efficient parameter set is identified:
  `n_paths = 100`, `n_sims = 100,000`, `n_steps = 252 × T` (one monitoring point per business day).

---

## 3. Repository contents

| Path | Description |
| --- | --- |
| `FE620_Final_Project.ipynb` | Final project notebook — full pricing engine **with executed outputs, plots and benchmark tables**. |
| `Pricer_v26.0.ipynb` | Release build of the same engine (clean source; identical sections/functions, no embedded outputs). |
| `report/FE620_Project_Report.pdf` | Full written report: theory, derivations, methodology, results & discussion. |
| `report/AsianOptions.pdf` | Reference / background material on Asian options. |
| `report/20.03.13_FE_620_Group_2_Asian_Options*.pdf` | Presentation slides and graded feedback. |

> The two notebooks share the same structure and functions. Use `Pricer_v26.0.ipynb` as the clean
> source of the algorithms, and `FE620_Final_Project.ipynb` to see the rendered results and figures.

---

## 4. Methods implemented

The notebooks are organised top-to-bottom into **Variables → Definitions → Performance Tests →
Hedging Test**. Each numerical routine is provided in up to three backends:

- **Conventional (NumPy)** — readable reference implementation.
- **JAX** — vectorised, JIT/GPU/TPU-accelerated path generation and pricing.
- **TensorFlow** — autodiff graph used to extract path-wise Greeks "for free".

### 4.1 GBM price-path generator
Simulates underlying price paths under risk-neutral GBM:

```
S_{t+Δt} = S_t · exp[(r − q − ½σ²)Δt + σ·ε·√Δt],   ε ~ N(0,1)
```

- `gbm_paths_original(S0, K, T, t, r, q, sigma, seed, n_paths, n_steps)` — NumPy cumulative-product paths.
- `gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)` — JAX version (fast, prepends `S0`, `cumprod`).
- `gbm_paths_hedging(...)` — variant for the hedging test where each path continues from the last price of the previous path (a continuous intraday trajectory).

### 4.2 Modified Black–Scholes for the geometric average (closed form)
Treats a geometric Asian option as a European option with **reduced volatility** \(\Sigma_G = \sigma/\sqrt{3}\)
and an adjusted forward \(G_0 = S_0\,e^{\frac{1}{2}(r-q)(T-t) - \frac{1}{12}\sigma^2(T-t)}\):

```python
def bsm_call(S0, K, T, t, r, q, sigma):
    G0 = S0 * np.exp(0.5*(r - q)*(T - t) - ((T - t)*sigma**2)/12)
    Sigma_G = sigma/np.sqrt(3)
    d1 = (np.log(G0/K) + 0.5*Sigma_G**2*(T - t)) / (Sigma_G*np.sqrt(T - t))
    d2 = (np.log(G0/K) - 0.5*Sigma_G**2*(T - t)) / (Sigma_G*np.sqrt(T - t))
    return np.exp(-r*(T - t)) * (G0*N(d1) - K*N(d2))
```

Functions: `bsm_call`, `bsm_put` (NumPy) and `bsm_call_tf`, `bsm_put_tf` (TensorFlow, with Greeks).

### 4.3 Lévy lognormal approximation (arithmetic average)
Approximates the arithmetic average as lognormal by matching the first two moments
\(\mu_{\text{lognormal}}, \sigma_{\text{lognormal}}\) of \(\int_0^T S_t\,dt\), then prices with a
Black–Scholes-style formula using forward \(F = S_0\,e^{\mu_{\text{lognormal}}}\). This is the fast
analytic estimate the project benchmarks against Monte Carlo.

### 4.4 Monte Carlo simulator
Repeatedly simulates GBM paths, averages each path (arithmetically or geometrically), computes the
discounted payoff, and averages across simulations.

- Arithmetic: `mc_call_arithm`, `mc_put_arithm` (JAX); `mc_call_arithm_np`, `mc_put_arithm_np` (NumPy).
- Geometric: `mc_call_geom`, `mc_put_geom` (JAX); `mc_call_geom_np`, `mc_put_geom_np` (NumPy).
- TensorFlow autodiff variants: `mc_call_arithm_tf`, `mc_call_geom_tf`, … (return price + Greeks).

```python
def mc_call_arithm(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims):
    c = []
    K = np.full(n_paths, K)
    payoff_0 = np.zeros(n_paths)
    for i in range(1, n_sims):
        S = gbm_paths_jax(S0, r, q, sigma, n_steps, n_paths)
        S_arithm = jnp.mean(S, axis=0)
        c.append(jnp.mean(jnp.exp(-r*T) * jnp.maximum(S_arithm - K, payoff_0)))
    return np.mean(c)
```

### 4.5 Variance reduction — control variates
Uses the **geometric** payoff (which has a known closed-form price) as a control variate for the
**arithmetic** payoff. The simulated difference `arithmetic − geometric` is corrected by the analytic
geometric price, producing **narrower 95% confidence intervals**. Functions:
`mc_call_control_variates`, `mc_put_control_variates` (NumPy) and `mc_call_cv`, `mc_put_cv` (JAX),
each returning the running estimate plus upper/lower confidence bounds.

### 4.6 Greeks
- **Analytic** first-/second-order Greeks for the BSM geometric model: `delta`, `gamma`, `vega`,
  `vomma`, `vanna`, `charm`, `theta`, `rho`, `phi`, … plus a sensitivity-surface plotter.
- **Automatic differentiation** (TensorFlow): nested `tf.gradients` calls produce a full Greeks table
  up to 3rd order (delta, vega, theta / gamma, vanna, charm / volga, veta / speed, zomma, color /
  ultima, totto).

### 4.7 Benchmarks & performance tests
- **Black–Scholes vs. Geometric Monte Carlo** — validates the MC engine against the closed form.
- **Linetsky test cases** — 7 arithmetic-average cases from Linetsky, *Exotic Spectra* (Risk, 2002),
  matched to within < 0.74% error.
- **Monte Carlo vs. number of trials** — log-error convergence vs. `n_paths` to choose efficient
  parameters.
- **Multiple control variates** — quantifies the variance-reduction benefit.

### 4.8 Hedging test
Generates fine-grained intraday paths (`n_steps` up to 1,000,000 ≈ 11 ticks/second) and compares the
**variability of a delta-hedged vs. unhedged** option portfolio over a simulated business week.

---

## 5. Model parameters

Set in the **Variables** section near the top of each notebook:

| Symbol | Variable | Meaning | Sample value |
| --- | --- | --- | --- |
| \(S_0\) | `S0` | Initial underlying price | 100 |
| \(r\) | `r` | Risk-free rate | 0.15 |
| \(q\) | `q` | Dividend yield | 0.0 |
| \(t\) | `t` | Valuation date | 0.0 |
| \(T\) | `T` | Maturity (years) | 1.0 |
| \(K\) | `K` | Strike | 100 |
| \(\sigma\) | `sigma` | Volatility | 0.3 |
| — | `n_paths` | Price paths per simulation | 100 |
| — | `n_steps` | Time steps per path | 504 (= 252 · T) |
| — | `n_sims` | Number of Monte Carlo simulations | 1000 |
| — | `seed`, `seed_0` | RNG seeds | 100 / 42 |

Plotting parameters (figure size, font sizes, colors) are configured immediately afterwards.

---

## 6. Getting started

### Requirements
- Python 3.10+ and Jupyter (the notebooks were authored in Google Colab; a `quant` kernel is fine).
- Packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `jax`, `numba`.
- Optional / for specific cells: `tensorflow` (autodiff Greeks), `quandl`, `yfinance`, `quantsbin`,
  `quantumrandom`.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install numpy pandas scipy matplotlib jax numba jupyter
# optional extras used by some cells
pip install tensorflow quandl yfinance quantsbin quantumrandom
```

> The notebooks contain commented `!pip install ...` lines (wrapped in triple quotes) intended for
> first-time Colab setup. Locally, install via the commands above instead.

### Run

```bash
jupyter notebook Pricer_v26.0.ipynb     # clean engine source
# or
jupyter notebook FE620_Final_Project.ipynb   # full version with outputs
```

Execute the cells top-to-bottom: **Python Packages → Variables → Definitions** (defines all pricing
functions), then any **Performance Tests** or the **Hedging Test**. For the TensorFlow cells use a
TF1-compatible environment (the code calls `tf.compat.v1` / `tf.placeholder`); a GPU/TPU runtime is
recommended for the autodiff Greeks. The JAX routines run well on CPU and scale on GPU/TPU.

---

## 7. Usage examples

After running the **Definitions** section, you can price options directly:

```python
# Closed-form geometric Asian (modified Black–Scholes)
geo_call = bsm_call(S0, K, T, t, r, q, sigma)

# Monte Carlo arithmetic Asian call
mc_call = mc_call_arithm(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims)

# Variance-reduced (control-variate) estimate with confidence bounds
c_cv, c_upper, c_lower = mc_call_cv(S0, K, T, t, r, q, sigma, n_paths, n_steps, n_sims)
```

Reproduce a Linetsky benchmark case:

```python
# Case 1:  r=0.02, sigma=0.10, T=1, S0=2.0, K=2.0, q=0
linetsky_ref = 0.0559860415
mc = mc_call_arithm(2.0, 2.0, 1, 0.0, 0.02, 0.0, 0.1, n_paths=100000, n_steps=252, n_sims=1000)
error_pct = abs((linetsky_ref - mc) / linetsky_ref) * 100
```

---

## 8. Key results

| Test | Outcome |
| --- | --- |
| Linetsky arithmetic-average cases (7) | Matched within **0.0147%–0.7345%** absolute error. |
| Lévy lognormal approximation | Fast but **imprecise** analytic estimate. |
| MC Black–Scholes + control variates | **Highly accurate**, at higher computation cost. |
| Optimal MC parameters | `n_paths = 100`, `n_sims = 100,000`, `n_steps = 252·T`. |

---

## 9. References

1. E. Lévy — *Pricing European average rate currency options*, Journal of International Money and
   Finance (1992).
2. A. Kemna & A. Vorst — *A pricing method for options based on average asset values* (1990).
3. V. Linetsky — *Exotic Spectra*, Risk Magazine, April 2002 (eigenfunction-expansion benchmarks).
4. Project report: `report/FE620_Project_Report.pdf`.

## Authors

Theo Dimitrasopoulos, William Kraemer, Snehal Rajguru, Vaikunth Seshadri — FE 620, Stevens Institute
of Technology (Advisor: Dan Pirjol).
