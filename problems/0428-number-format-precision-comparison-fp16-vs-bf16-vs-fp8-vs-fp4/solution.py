import math
import numpy as np

def compare_formats(values):
    """
    Simulate quantizing FP32 values to FP16, BF16, FP8_E4M3, and FP4_E2M1 formats.
    """
    formats = {
        'fp16':     (5, 10, True),
        'bf16':     (8, 7,  True),
        'fp8_e4m3': (4, 3,  False),
        'fp4_e2m1': (2, 1,  False),
    }

    vals = [float(v) for v in values]
    result = {}

    for name, (e, m, has_inf) in formats.items():
        bias = (1 << (e - 1)) - 1
        min_normal = 2.0 ** (1 - bias)
        if has_inf:
            max_val = (2.0 ** ((1 << e) - 2 - bias)) * (2.0 - 2.0 ** (-m))
        else:
            max_val = (2.0 ** ((1 << e) - 1 - bias)) * (2.0 - 2.0 ** (1 - m))

        subnormal_step = 2.0 ** (1 - bias - m)
        quantized = []
        errors = []

        for v in vals:
            if v == 0.0 or math.isnan(v):
                q = 0.0
            else:
                sign = -1.0 if v < 0 else 1.0
                abs_v = abs(v)

                if abs_v > max_val:
                    q = sign * float('inf') if has_inf else sign * max_val
                else:
                    if abs_v < min_normal:
                        step = subnormal_step
                    else:
                        E = math.floor(math.log2(abs_v))
                        step = 2.0 ** (E - m)

                    q_mag = round(abs_v / step) * step
                    q = 0.0 if q_mag == 0.0 else sign * q_mag

            quantized.append(q)

            if math.isinf(q) and math.isfinite(v):
                errors.append(float('inf'))
            elif math.isnan(v):
                errors.append(0.0)
            else:
                errors.append(abs(v - q))

        max_err = max(errors) if errors else 0.0
        mean_err = (sum(errors) / len(errors)) if errors else 0.0

        result[name] = {
            'max_representable': max_val,
            'min_positive_normal': min_normal,
            'quantized': quantized,
            'max_abs_error': round(max_err, 6) if math.isfinite(max_err) else float('inf'),
            'mean_abs_error': round(mean_err, 6) if math.isfinite(mean_err) else float('inf'),
        }

    return result