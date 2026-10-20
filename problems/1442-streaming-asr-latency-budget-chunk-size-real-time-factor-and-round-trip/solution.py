import math


def chunk_latency(chunk_s, rtf, rtt_s):
    return chunk_s + rtf * chunk_s + rtt_s


def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    if chunk_latency(step, rtf, rtt_s) > budget_s + 1e-9:
        return None

    multiple = 1
    while chunk_latency(round((multiple + 1) * step, 9), rtf, rtt_s) <= budget_s + 1e-9:
        multiple += 1
    
    return round(step * multiple, 2)

def concurrent_streams(rtf, gpu_share=1.0):
    return math.floor(gpu_share / rtf)


def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)


def budget_breakdown(chunk_s, rtf, rtt_s):
    # TODO: {'wait', 'model', 'network'} fractions rounded to 4 decimals
    total = chunk_latency(chunk_s, rtf, rtt_s)
    if total == 0:
        return {'wait': 0.0, 'model': 0.0, 'network': 0.0}
        
    return {
        'wait': round(chunk_s / total, 4),
        'model': round((rtf * chunk_s) / total, 4),
        'network': round(rtt_s / total, 4),
    }
