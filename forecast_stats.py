"""
Turns a set of Monte-Carlo forecast paths into distribution statistics.

Kronos is a stochastic sampler: drawing it K times gives K plausible futures.
The spread across those futures is the model's own uncertainty, and it is what
position sizing needs. A +3% forecast that every path agrees on is a very
different bet from a +3% forecast averaged out of paths ranging -8% to +14%.

Pure functions only — no model, no network, no config.
"""
import numpy as np

# Keeps conviction finite when every path lands on an identical value.
SIGMA_FLOOR = 1e-6


def summarize(returns) -> dict | None:
    """Summarize terminal returns (decimals, 0.03 == +3%) across sample paths.

    Returns None when fewer than two finite values are available, because
    dispersion — and therefore conviction — is undefined for a single path.
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None

    mu = float(arr.mean())
    sigma = max(float(arr.std(ddof=1)), SIGMA_FLOOR)
    ir = mu / sigma

    return {
        "mu": mu,
        "sigma": sigma,
        "p_up": float((arr > 0).mean()),
        "ir": ir,
        "conviction": abs(ir),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "n_paths": int(arr.size),
    }
