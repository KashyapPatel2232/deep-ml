import numpy as np


def _quantize_1d_symmetric(vec: np.ndarray, bits: int = 8) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    q_max = (1 << (bits - 1)) - 1
    max_val = float(np.max(np.abs(v)))
    if max_val == 0.0:
        return np.zeros_like(vec)
    scale = max_val / q_max
    q = np.clip(np.round(v / scale), -q_max, q_max) * scale
    return q.astype(vec.dtype)


def _gptq_quantize_linear(
    W: np.ndarray,
    H_base: np.ndarray,
    bits: int = 8,
    block_size: int = 64,
    damp_percent: float = 0.01,
    actorder: bool = True,
) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64).copy()
    H = H_base.astype(np.float64, copy=True)
    d_in, _ = W.shape
    q_max = (1 << (bits - 1)) - 1

    max_val = float(np.max(np.abs(W)))
    if max_val == 0.0:
        return np.zeros_like(W)
    scale = max_val / q_max

    diag_mean = float(np.mean(np.diag(H)))
    damp = damp_percent * diag_mean if diag_mean > 0.0 else 1e-5
    idx_diag = np.arange(d_in)
    H[idx_diag, idx_diag] += damp

    if actorder:
        perm = np.argsort(np.diag(H))[::-1]
        inv_perm = np.argsort(perm)
        W = W[perm, :]
        H = H[np.ix_(perm, perm)]

    try:
        L = np.linalg.cholesky(H)
        L_inv = np.linalg.solve(L, np.eye(d_in))
        U = np.linalg.cholesky(L_inv.T @ L_inv).T
    except np.linalg.LinAlgError:
        H[idx_diag, idx_diag] += max(1e-2, 0.1 * diag_mean)
        L = np.linalg.cholesky(H)
        L_inv = np.linalg.solve(L, np.eye(d_in))
        U = np.linalg.cholesky(L_inv.T @ L_inv).T

    Q = np.zeros_like(W)
    for i in range(0, d_in, block_size):
        i_end = min(i + block_size, d_in)
        count = i_end - i

        W_block = W[i:i_end, :].copy()
        Q_block = np.zeros_like(W_block)
        E_block = np.zeros_like(W_block)
        U_block = U[i:i_end, i:i_end]

        for j in range(count):
            w = W_block[j, :]
            d = U_block[j, j]

            q = np.clip(np.round(w / scale), -q_max, q_max) * scale
            Q_block[j, :] = q

            err = (w - q) / d
            E_block[j, :] = err
            if j + 1 < count:
                W_block[j + 1 :, :] -= np.outer(U_block[j, j + 1 :], err)

        Q[i:i_end, :] = Q_block

        if i_end < d_in:
            W[i_end:, :] -= U[i:i_end, i_end:].T @ E_block

    if actorder:
        Q = Q[inv_perm, :]

    return Q


def ptq(weights, calib_X):
    dequant_weights = weights.copy()

    X_fp32 = np.asarray(calib_X, dtype=np.float64)
    if X_fp32.ndim == 2 and X_fp32.shape[0] == 784 and X_fp32.shape[1] != 784:
        X_fp32 = X_fp32.T
    X_quant = X_fp32.copy()
    n_samples = max(X_fp32.shape[0], 1)

    w_keys = [k for k, v in weights.items() if np.asarray(v).ndim == 2]
    b_keys = [k for k, v in weights.items() if np.asarray(v).ndim == 1]

    for idx, w_key in enumerate(w_keys):
        W_orig = np.asarray(weights[w_key])
        d_in = X_quant.shape[1]

        transposed = W_orig.shape[0] != d_in and W_orig.shape[1] == d_in
        W_work = (W_orig.T if transposed else W_orig).astype(np.float64)

        has_bias = idx < len(b_keys)
        b_raw = np.asarray(weights[b_keys[idx]]) if has_bias else None
        b_orig = b_raw.astype(np.float64) if has_bias else 0.0

        Y_fp32 = X_fp32 @ W_work + b_orig
        H_base = (2.0 / n_samples) * (X_quant.T @ X_quant)

        W_target = W_work
        if idx > 0:
            diag_mean = float(np.mean(np.diag(H_base)))
            cross_damp = max(0.05, d_in / (2.0 * n_samples)) * (diag_mean if diag_mean > 0.0 else 1e-5)
            H_reg = H_base.copy()
            H_reg[np.arange(d_in), np.arange(d_in)] += cross_damp

            drift_rhs = (2.0 / n_samples) * (X_quant.T @ ((X_fp32 - X_quant) @ W_work))
            dW_cross = np.linalg.solve(H_reg, drift_rhs)
            W_target = W_work + dW_cross

        W_hat_work = _gptq_quantize_linear(
            W_target,
            H_base,
            bits=8,
            block_size=64,
            damp_percent=0.01,
            actorder=True,
        ).astype(W_orig.dtype)
        dequant_weights[w_key] = W_hat_work.T if transposed else W_hat_work

        b_hat = 0.0
        if has_bias:
            mean_residual = np.mean(Y_fp32 - (X_quant @ W_hat_work.astype(np.float64) + b_orig), axis=0)
            b_comp = (b_orig + mean_residual).astype(b_raw.dtype)
            b_hat = _quantize_1d_symmetric(b_comp, bits=8)
            dequant_weights[b_keys[idx]] = b_hat

        if idx < len(w_keys) - 1:
            X_fp32 = np.maximum(0.0, Y_fp32)
            X_quant = np.maximum(0.0, X_quant @ W_hat_work.astype(np.float64) + np.asarray(b_hat, dtype=np.float64))

    return dequant_weights