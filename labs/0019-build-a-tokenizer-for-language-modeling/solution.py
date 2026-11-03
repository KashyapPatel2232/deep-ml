import math
from collections import Counter, defaultdict

def train_tokenizer(corpus, vocab_size):
    chars = sorted(set("".join(corpus)))
    num_chars = len(chars)
    if vocab_size <= num_chars:
        c2i = {c: i for i, c in enumerate(chars[:vocab_size])}
        i2c = {i: c for c, i in c2i.items()}
        return (lambda s: [c2i[c] for c in s]), (lambda ids: "".join(i2c[t] for t in ids))

    word_counts = Counter(corpus)
    unique_words = list(word_counts.keys())
    freqs = [word_counts[w] for w in unique_words]
    total_words = sum(freqs)

    bpe_t2i = {c: i for i, c in enumerate(chars)}
    bpe_i2t = {i: c for i, c in enumerate(chars)}
    bpe_words = [[bpe_t2i[c] for c in w] for w in unique_words]

    max_bpe_merges = min(300, max(vocab_size - num_chars + 80, vocab_size))
    for _ in range(max_bpe_merges):
        pair_counts = defaultdict(int)
        for w, f in zip(bpe_words, freqs):
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += f
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        if pair_counts[best_pair] < 10:
            break
        new_id = len(bpe_t2i)
        new_str = bpe_i2t[best_pair[0]] + bpe_i2t[best_pair[1]]
        bpe_t2i[new_str] = new_id
        bpe_i2t[new_id] = new_str

        p0, p1 = best_pair
        for idx, w in enumerate(bpe_words):
            if len(w) < 2:
                continue
            new_w = []
            i = 0
            n = len(w)
            while i < n:
                if i < n - 1 and w[i] == p0 and w[i + 1] == p1:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            bpe_words[idx] = new_w

    sub_total = defaultdict(int)
    sub_prefix = defaultdict(int)
    sub_suffix = defaultdict(int)
    sub_whole = defaultdict(int)

    for w, f in zip(unique_words, freqs):
        n = len(w)
        for i in range(n):
            max_j = min(n, i + 6)
            for j in range(i + 2, max_j + 1):
                s = w[i:j]
                sub_total[s] += f
                if i == 0 and j == n:
                    sub_whole[s] += f
                elif i == 0:
                    sub_prefix[s] += f
                elif j == n:
                    sub_suffix[s] += f

    scored_subs = []
    for s, tot in sub_total.items():
        if tot < 12:
            continue

        weighted_freq = (
            tot
            + 0.35 * sub_prefix[s]
            + 0.55 * sub_suffix[s]
            + 0.90 * sub_whole[s]
        )
        score = weighted_freq * (len(s) - 0.85)
        scored_subs.append((score, s))

    scored_subs.sort(reverse=True)

    candidate_set = set(chars)
    candidate_list = list(chars)

    target_pool_size = max(vocab_size + 160, 420)
    for tok_id in range(num_chars, len(bpe_i2t)):
        s = bpe_i2t[tok_id]
        if len(s) <= 6 and sub_total[s] >= 12 and s not in candidate_set:
            candidate_set.add(s)
            candidate_list.append(s)

    for _, s in scored_subs:
        if len(candidate_list) >= target_pool_size:
            break
        if s not in candidate_set:
            candidate_set.add(s)
            candidate_list.append(s)

    
    cand_t2i = {s: i for i, s in enumerate(candidate_list)}
    num_cands = len(candidate_list)

    word_spans = []
    for w in unique_words:
        n = len(w)
        spans_at_end = [[] for _ in range(n + 1)]
        for i in range(1, n + 1):
            start_min = max(0, i - 6)
            for j in range(start_min, i):
                tid = cand_t2i.get(w[j:i])
                if tid is not None:
                    spans_at_end[i].append((j, tid))
        word_spans.append(spans_at_end)

    active = [True] * num_cands

    uni_cost = [3.0 if len(s) == 1 else 6.5 - 0.4 * len(s) for s in candidate_list]
    bos_cost = list(uni_cost)
    eos_cost = [2.0] * num_cands

    word_segs = [[] for _ in range(len(unique_words))]

    for _ in range(2):
        tok_counts = [0] * num_cands
        bos_counts = [0] * num_cands
        eos_counts = [0] * num_cands
        total_toks = 0

        for w_idx, spans_at_end in enumerate(word_spans):
            n = len(spans_at_end) - 1
            dp_cost = [0.0] + [1e18] * n
            dp_prev = [0] * (n + 1)
            dp_tok = [0] * (n + 1)

            for i in range(1, n + 1):
                best_c = 1e18
                best_j = 0
                best_t = 0
                is_end = (i == n)
                for j, tid in spans_at_end[i]:
                    if not active[tid]:
                        continue
                    c = dp_cost[j] + (bos_cost[tid] if j == 0 else uni_cost[tid])
                    if is_end:
                        c += eos_cost[tid]
                    if c < best_c:
                        best_c = c
                        best_j = j
                        best_t = tid
                dp_cost[i] = best_c
                dp_prev[i] = best_j
                dp_tok[i] = best_t

            seq = []
            curr = n
            while curr > 0:
                seq.append(dp_tok[curr])
                curr = dp_prev[curr]
            seq.reverse()
            word_segs[w_idx] = seq

            f = freqs[w_idx]
            total_toks += f * len(seq)
            bos_counts[seq[0]] += f
            eos_counts[seq[-1]] += f
            for tid in seq:
                tok_counts[tid] += f

        denom_uni = total_toks + vocab_size
        denom_bos = total_words + vocab_size
        for tid in range(num_cands):
            if active[tid]:
                uni_cost[tid] = -math.log2((tok_counts[tid] + 1.0) / denom_uni) + 1.15
                bos_cost[tid] = 0.55 * (-math.log2((bos_counts[tid] + 1.0) / denom_bos)) + 0.45 * uni_cost[tid]
                eos_cost[tid] = -math.log2((eos_counts[tid] + 1.0) / (tok_counts[tid] + vocab_size + 1.0))

    
    BOS_IDX = num_cands
    smoothing_k = float(vocab_size + 1)

    def build_bigram_costs(segs, loo_discount=0.35):
        row_counts = [0] * (num_cands + 1)
        pair_counts = defaultdict(int)

        for w_idx, seq in enumerate(segs):
            f = freqs[w_idx]
            prev = BOS_IDX
            for tid in seq:
                pair_counts[(prev, tid)] += f
                row_counts[prev] += f
                prev = tid
            pair_counts[(prev, BOS_IDX)] += f
            row_counts[prev] += f

        default_row_cost = [
            math.log2(rc + smoothing_k) for rc in row_counts
        ]
        trans_cost = {}
        for (u, v), c in pair_counts.items():
            eff_c = max(0.0, c - loo_discount) if loo_discount > 0.0 else float(c)
            trans_cost[(u, v)] = -math.log2((eff_c + 1.0) / (row_counts[u] + smoothing_k))

        return trans_cost, default_row_cost, row_counts

    def viterbi_segment_corpus(trans_cost, default_row_cost):
        new_segs = [[] for _ in range(len(unique_words))]
        for w_idx, spans_at_end in enumerate(word_spans):
            n = len(spans_at_end) - 1
            dp = [[] for _ in range(n + 1)]
            for i in range(1, n + 1):
                entries = []
                for j, tid in spans_at_end[i]:
                    if not active[tid]:
                        continue
                    if j == 0:
                        c = trans_cost.get((BOS_IDX, tid), default_row_cost[BOS_IDX])
                        entries.append((tid, c, 0, -1))
                    else:
                        best_c = 1e18
                        best_k = -1
                        for k_idx, (prev_tid, prev_c, _, _) in enumerate(dp[j]):
                            tc = prev_c + trans_cost.get((prev_tid, tid), default_row_cost[prev_tid])
                            if tc < best_c:
                                best_c = tc
                                best_k = k_idx
                        entries.append((tid, best_c, j, best_k))
                dp[i] = entries

            best_final_c = 1e18
            best_last_idx = 0
            for k_idx, (tid, c, _, _) in enumerate(dp[n]):
                fc = c + trans_cost.get((tid, BOS_IDX), default_row_cost[tid])
                if fc < best_final_c:
                    best_final_c = fc
                    best_last_idx = k_idx

            seq = []
            curr_i = n
            curr_k = best_last_idx
            while curr_i > 0:
                tid, _, prev_j, prev_k = dp[curr_i][curr_k]
                seq.append(tid)
                curr_i = prev_j
                curr_k = prev_k
            seq.reverse()
            new_segs[w_idx] = seq
        return new_segs

    def fallback_decomposition(tid, trans_cost, default_row_cost):
        s = candidate_list[tid]
        m = len(s)
        dp_c = [0.0] + [1e18] * m
        dp_p = [0] * (m + 1)
        dp_t = [0] * (m + 1)
        for i in range(1, m + 1):
            for j in range(0, i):
                if j == 0 and i == m:
                    continue
                sub_id = cand_t2i.get(s[j:i])
                if sub_id is not None and active[sub_id]:
                    step_c = uni_cost[sub_id]
                    if dp_c[j] + step_c < dp_c[i]:
                        dp_c[i] = dp_c[j] + step_c
                        dp_p[i] = j
                        dp_t[i] = sub_id
        seq = []
        curr = m
        while curr > 0:
            seq.append(dp_t[curr])
            curr = dp_p[curr]
        seq.reverse()
        internal_c = 0.0
        for q in range(len(seq) - 1):
            u, v = seq[q], seq[q + 1]
            internal_c += trans_cost.get((u, v), default_row_cost[u])
        return seq[0], seq[-1], internal_c

    curr_active_count = sum(active)
    prune_targets = [
        t for t in [350, 300, 272, vocab_size] if vocab_size <= t < curr_active_count
    ]
    if not prune_targets or prune_targets[-1] != vocab_size:
        prune_targets.append(vocab_size)

    for target_sz in prune_targets:
        trans_cost, default_row_cost, row_counts = build_bigram_costs(word_segs, loo_discount=0.35)
        word_segs = viterbi_segment_corpus(trans_cost, default_row_cost)
        trans_cost, default_row_cost, row_counts = build_bigram_costs(word_segs, loo_discount=0.25)

        if sum(active) <= target_sz:
            continue

        
        fb_first = [0] * num_cands
        fb_last = [0] * num_cands
        fb_inter = [0.0] * num_cands
        for tid in range(num_chars, num_cands):
            if active[tid]:
                u_f, u_l, inter_c = fallback_decomposition(tid, trans_cost, default_row_cost)
                fb_first[tid] = u_f
                fb_last[tid] = u_l
                fb_inter[tid] = inter_c

        
        delta_bits = [0.0] * num_cands
        for w_idx, seq in enumerate(word_segs):
            f = freqs[w_idx]
            k_len = len(seq)
            for r, tid in enumerate(seq):
                if tid < num_chars:
                    continue
                left = BOS_IDX if r == 0 else seq[r - 1]
                right = BOS_IDX if r == k_len - 1 else seq[r + 1]

                c_with = (
                    trans_cost.get((left, tid), default_row_cost[left])
                    + trans_cost.get((tid, right), default_row_cost[tid])
                )
                u_f = fb_first[tid]
                u_l = fb_last[tid]
                c_without = (
                    trans_cost.get((left, u_f), default_row_cost[left])
                    + fb_inter[tid]
                    + trans_cost.get((u_l, right), default_row_cost[u_l])
                )
                diff = c_without - c_with
                if diff > 0.0:
                    delta_bits[tid] += f * diff

        
        multi_candidates = [
            (delta_bits[tid], row_counts[tid], tid)
            for tid in range(num_chars, num_cands)
            if active[tid]
        ]
        multi_candidates.sort(reverse=True)
        keep_multi = set(tid for _, _, tid in multi_candidates[: target_sz - num_chars])

        for tid in range(num_chars, num_cands):
            if active[tid] and tid not in keep_multi:
                active[tid] = False

    
    for discount in (0.35, 0.20):
        trans_cost, default_row_cost, _ = build_bigram_costs(word_segs, loo_discount=discount)
        word_segs = viterbi_segment_corpus(trans_cost, default_row_cost)

    
    final_tokens = [candidate_list[tid] for tid in range(num_cands) if active[tid]][:vocab_size]
    token_to_id = {s: i for i, s in enumerate(final_tokens)}
    id_to_token = {i: s for i, s in enumerate(final_tokens)}
    old_to_new = {cand_t2i[s]: i for i, s in enumerate(final_tokens)}

    V = len(final_tokens)
    FINAL_BOS = V
    max_tok_len = max(len(s) for s in final_tokens)


    final_row_counts = [0] * (V + 1)
    final_pair_counts = [[0] * (V + 1) for _ in range(V + 1)]
    cache = {}

    for w_idx, w in enumerate(unique_words):
        f = freqs[w_idx]
        mapped_seq = [old_to_new[tid] for tid in word_segs[w_idx]]
        cache[w] = mapped_seq
        prev = FINAL_BOS
        for tid in mapped_seq:
            final_pair_counts[prev][tid] += f
            final_row_counts[prev] += f
            prev = tid
        final_pair_counts[prev][FINAL_BOS] += f
        final_row_counts[prev] += f

    
    final_uni_prior = [
        -0.015 * math.log2((final_row_counts[t] + 1.0) / (total_words + V))
        for t in range(V + 1)
    ]
    cost_matrix = [[0.0] * (V + 1) for _ in range(V + 1)]
    for u in range(V + 1):
        denom = final_row_counts[u] + float(vocab_size + 1)
        row_u = final_pair_counts[u]
        cost_u = cost_matrix[u]
        for v in range(V + 1):
            c = row_u[v]
            cost_u[v] = -math.log2((c + 1.0) / denom) + (final_uni_prior[v] if c == 0 else 0.0)

    def encode(text):
        if not text:
            return []
        if text in cache:
            return list(cache[text])

        n = len(text)
        dp = [[] for _ in range(n + 1)]
        bos_row = cost_matrix[FINAL_BOS]

        for i in range(1, n + 1):
            entries = []
            start_min = max(0, i - max_tok_len)
            for j in range(start_min, i):
                tid = token_to_id.get(text[j:i])
                if tid is None:
                    continue
                if j == 0:
                    entries.append((tid, bos_row[tid], 0, -1))
                else:
                    best_c = 1e18
                    best_k = -1
                    for k_idx, (prev_tid, prev_c, _, _) in enumerate(dp[j]):
                        tc = prev_c + cost_matrix[prev_tid][tid]
                        if tc < best_c:
                            best_c = tc
                            best_k = k_idx
                    entries.append((tid, best_c, j, best_k))
            dp[i] = entries

        best_final_c = 1e18
        best_last_idx = 0
        for k_idx, (tid, c, _, _) in enumerate(dp[n]):
            fc = c + cost_matrix[tid][FINAL_BOS]
            if fc < best_final_c:
                best_final_c = fc
                best_last_idx = k_idx

        res = []
        curr_i = n
        curr_k = best_last_idx
        while curr_i > 0:
            tid, _, prev_j, prev_k = dp[curr_i][curr_k]
            res.append(tid)
            curr_i = prev_j
            curr_k = prev_k
        res.reverse()

        cache[text] = res
        return list(res)

    def decode(token_ids):
        return "".join(id_to_token[t] for t in token_ids)

    return encode, decode