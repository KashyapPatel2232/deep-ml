import math
import torch


def compare_formats(values) -> dict:
    """
    Simulate quantizing FP32 values to FP16, BF16, FP8_E4M3, and FP4_E2M1 formats
    using PyTorch's native dtype casting and tensor operations.
    """
    t = torch.as_tensor(values, dtype=torch.float32).flatten()
    is_nan = torch.isnan(t)
    clean = torch.where(is_nan | (t == 0.0), torch.zeros_like(t), t)

    formats = {
        'fp16':     (5, 10, True,  torch.float16),
        'bf16':     (8, 7,  True,  torch.bfloat16),
        'fp8_e4m3': (4, 3,  False, None),
        'fp4_e2m1': (2, 1,  False, None),
    }

    result = {}

    for name, (e, m, has_inf, native_dtype) in formats.items():
        bias = (1 << (e - 1)) - 1
        min_normal = 2.0 ** (1 - bias)
        if has_inf:
            max_val = (2.0 ** ((1 << e) - 2 - bias)) * (2.0 - 2.0 ** (-m))
        else:
            max_val = (2.0 ** ((1 << e) - 1 - bias)) * (2.0 - 2.0 ** (1 - m))

        if native_dtype is not None:
            q = clean.to(native_dtype).to(torch.float32)
            q = torch.where(q == 0.0, torch.zeros_like(q), q)
        else:
            subnormal_step = 2.0 ** (1 - bias - m)
            sign = torch.sign(clean)
            clamped = torch.clamp(clean.abs(), max=max_val)

            safe_abs = torch.where(clamped > 0.0, clamped, torch.ones_like(clamped))
            E = torch.floor(torch.log2(safe_abs))
            normal_step = torch.pow(2.0, E - m)
            step = torch.where(clamped < min_normal, torch.full_like(clamped, subnormal_step), normal_step)

            q_mag = torch.round(clamped / step) * step
            q = torch.where(q_mag == 0.0, torch.zeros_like(q_mag), sign * q_mag)

        abs_err = torch.where(
            is_nan,
            torch.zeros_like(t),
            torch.where(
                torch.isinf(q) & torch.isfinite(t),
                torch.full_like(t, float('inf')),
                (t - q).abs(),
            ),
        )

        max_err = abs_err.max().item() if abs_err.numel() > 0 else 0.0
        mean_err = abs_err.mean().item() if abs_err.numel() > 0 else 0.0

        result[name] = {
            'max_representable': max_val,
            'min_positive_normal': min_normal,
            'quantized': q.tolist(),
            'max_abs_error': round(max_err, 6) if math.isfinite(max_err) else float('inf'),
            'mean_abs_error': round(mean_err, 6) if math.isfinite(mean_err) else float('inf'),
        }

    return result