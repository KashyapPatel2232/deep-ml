import numpy as np

def simplified_gptq(w: np.ndarray, H_diag: np.ndarray, bits: int = 4, blocksize: int = 128):
    """
    Simplified GPTQ: sequential symmetric quant with within-block
    diagonal-Hessian error compensation.

    scale = max|w| / qmax once for the row (qmax=2^(bits-1)-1).
    For each i in each block:
      q_i = clamp(round(w_b[i]/scale), -qmax, qmax)
      w_hat_i = q_i * scale
      e = w_b[i] - w_hat_i
      for j > i in block: w_b[j] -= e * H_diag[i] / (H_diag[j] + 1e-8)

    Returns q (int n,), scale (float), w_hat (float n,)
    """
    # Your code here
    w = np.asarray(w, dtype=float)
    H_diag = np.asarray(H_diag, dtype=float)
    n = len(w)

    qmax = 2 ** (bits - 1) - 1
    max_val = np.max(np.abs(w))
    scale = float(max_val / qmax) if max_val > 0 else 1.0

    w_b = w.copy()
    q = np.zeros(n, dtype = int)
    w_hat = np.zeros(n, dtype = float)
    
    for start in range(0, n, blocksize):
      end = min(start + blocksize, n)

      for i in range(start, end):
        q[i] = int(np.clip(np.round(w_b[i]/scale), -qmax, qmax))
        w_hat[i] = q[i] * scale
        e = w_b[i] - w_hat[i]
        if i + 1 < end:
          w_b[i+1:end] -= e * H_diag[i] / (H_diag[i+1 :end] + 1e-8)

    return q, scale, w_hat
