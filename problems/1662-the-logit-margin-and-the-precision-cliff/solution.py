import numpy as np


def logit_margin(z, t):
  """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
  z_arr = np.asarray(z, dtype=np.float64)

  z_other = np.delete(z_arr, t)

  max_val = np.max(z_other)
  log_sum_exp = max_val + np.log(np.sum(np.exp(z_other - max_val)))

  return float(z_arr[t] - log_sum_exp)


def precision_cliff(dtype):
  """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype;

  bisection on a float64 m. Returns the margin rounded to 4 decimals.
  """
  dt = np.dtype(dtype)
  low = 0.0
  high = 100.0

  while (high - low) > 1e-9:
    mid = (low + high) / 2.0
    exp_term = dt.type(np.exp(-mid))
    res = 1.0 / (dt.type(1.0) + exp_term)

    if res == dt.type(1.0):
      high = mid
    else:
      low = mid

  return round(float(high), 4)