import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

def _exp_no_offset(x, a, b):
    return a * np.exp(b * x)

def _exp_with_offset(x, a, b, c):
    return a * np.exp(b * x) + c

def _metrics(y_true, y_pred, k):
    # k = number of fitted parameters
    resid = y_true - y_pred
    n = len(y_true)
    sse = np.sum(resid**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - sse / sst if sst > 0 else 0.0
    rmse = np.sqrt(sse / n)
    # AIC/BIC for Gaussian residuals with unknown variance
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return {"r2": r2, "rmse": rmse, "aic": aic, "bic": bic, "sse": sse, "resid": resid}

def exponential_filter(x, y, r2_thresh=0.95, rmse_frac_thresh=0.1, try_offset=True):
    """
    Decide if (x,y) follow an exponential relationship.
    Returns a dict with decision, parameters, and diagnostics.

    r2_thresh: minimum R^2 to accept
    rmse_frac_thresh: RMSE as fraction of data span allowed (lower is better)
    try_offset: also try y = a*exp(bx)+c if basic model struggles (e.g., y<=0)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    assert x.shape == y.shape and x.ndim == 1 and len(x) >= 5, "Use 1D arrays with >=5 points"

    # sort by x to
