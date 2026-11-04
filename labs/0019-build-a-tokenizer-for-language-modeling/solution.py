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
    total_chars = sum(len(w) * f for w, f in zip(unique_words, freqs))

    bpe_t2i = {c: i for i, c in enumerate(chars)}
    bpe_i2t = {i: c for i, c in enumerate(chars)}
    bpe_words = [[bpe_t2i[c] for c in w] for w in unique_words]

    max_bpe_merges = min(220, max(vocab_size - num_chars, 180))
    for _ in range(max_bpe_merges):
        pair_scores = defaultdict(float)
        pair_raw = defaultdict(int)
        for w, f in zip(bpe_words, freqs):
            n = len(w)
            if n < 2:
                continue
            for i in range(n - 1):
                pair = (w[i], w[i + 1])
                pair_raw[pair] += f
                if n == 2:
                    pair_scores[pair] += 1.50 * f
                elif i == n - 2:
                    pair_scores[pair] += 1.35 * f
                elif i == 0:
                    pair_scores[pair] += 1.20 * f
                else:
                    pair_scores[pair] += float(f)

        if not pair_scores:
            break
        best_pair = max(pair_scores, key=lambda p: (pair_scores[p], -p[0], -p[1]))
        if pair_raw[best_pair] < 10:
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
    sub_internal = defaultdict(int)
    sub_whole = defaultdict(int)

    for w, f in zip(unique_words, freqs):
        n = len(w)
        for i in range(n):
            max_j = min(n, i + 6)
            for j in range(i + 1, max_j + 1):
                s = w[i:j]
                sub_total[s] += f
                if i == 0 and j == n:
                    sub_whole[s] += f
                elif i == 0:
                    sub_prefix[s] += f
                elif j == n:
                    sub_suffix[s] += f
                else:
                    sub_internal[s] += f

    candidate_set = set(chars)
    candidate_list = list(chars)

    for tok_id in range(num_chars, len(bpe_i2t)):
        s = bpe_i2t[tok_id]
        if len(s) <= 6 and sub_total[s] >= 12 and s not in candidate_set:
            candidate_set.add(s)
            candidate_list.append(s)

    scored_subs = []
    for s, tot in sub_total.items():
        if len(s) < 2 or tot < 12 or s in candidate_set:
            continue
        end_c = sub_suffix[s] + sub_whole[s]
        start_c = sub_prefix[s] + sub_whole[s]
        end_purity = end_c / tot
        start_purity = start_c / tot
        purity_bonus = 1.0 + 0.85 * (end_purity ** 2) + 0.50 * (start_purity ** 2)
        len_bonus = (len(s) - 0.72) if len(s) <= 4 else (3.28 + 0.45 * (len(s) - 4))
        score = tot * purity_bonus * len_bonus
        scored_subs.append((score, s))

    scored_subs.sort(reverse=True)
    target_pool_size = max(vocab_size + 165, 420)
    for _, s in scored_subs:
        if len(candidate_list) >= target_pool_size:
            break
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

    bos_cost = [0.0] * num_cands
    uni_cost = [0.0] * num_cands
    eos_cost = [0.0] * num_cands
    cont_cost = [0.0] * num_cands

    denom_start_init = total_words + num_cands
    denom_mid_init = total_chars + num_cands
    for tid, s in enumerate(candidate_list):
        tot = sub_total[s]
        st = sub_prefix[s] + sub_whole[s]
        md = sub_internal[s] + sub_suffix[s]
        ed = sub_suffix[s] + sub_whole[s]
        ct = tot - ed
        init_pen = 4.2 if len(s) == 1 else max(0.0, 0.50 * (3 - len(s)))
        bos_cost[tid] = -math.log2((st + 1.0) / denom_start_init) + init_pen
        uni_cost[tid] = -math.log2((md + 1.0) / denom_mid_init) + init_pen
        eos_cost[tid] = -math.log2((ed + 1.0) / (tot + 2.0))
        cont_cost[tid] = -math.log2((ct + 1.0) / (tot + 2.0))

    word_segs = [[] for _ in range(len(unique_words))]

    for em_iter in range(4):
        start_counts = [0] * num_cands
        mid_counts = [0] * num_cands
        end_counts = [0] * num_cands
        cont_counts = [0] * num_cands
        total_mid = 0
        token_penalty = 0.55 if em_iter < 2 else 0.30

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
                    c = (
                        dp_cost[j]
                        + (bos_cost[tid] if j == 0 else uni_cost[tid])
                        + (eos_cost[tid] if is_end else cont_cost[tid])
                        + token_penalty
                    )
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
            k_len = len(seq)
            start_counts[seq[0]] += f
            end_counts[seq[-1]] += f
            for r in range(k_len):
                tid = seq[r]
                if r > 0:
                    mid_counts[tid] += f
                    total_mid += f
                if r < k_len - 1:
                    cont_counts[tid] += f

        denom_start = total_words + vocab_size
        denom_mid = max(1, total_mid) + vocab_size
        char_fb = 1.15 if em_iter < 2 else 0.65
        for tid in range(num_cands):
            tot_t = end_counts[tid] + cont_counts[tid]
            pen = char_fb if tid < num_chars else 0.0
            bos_cost[tid] = -math.log2((start_counts[tid] + 1.0) / denom_start) + pen
            uni_cost[tid] = -math.log2((mid_counts[tid] + 1.0) / denom_mid) + pen
            eos_cost[tid] = -math.log2((end_counts[tid] + 1.0) / (tot_t + 2.0))
            cont_cost[tid] = -math.log2((cont_counts[tid] + 1.0) / (tot_t + 2.0))

    discovered_merges = defaultdict(int)
    for w_idx, seq in enumerate(word_segs):
        f = freqs[w_idx]
        for r in range(len(seq) - 1):
            s = candidate_list[seq[r]] + candidate_list[seq[r + 1]]
            if len(s) <= 6 and s not in candidate_set:
                discovered_merges[s] += f

    top_discovered = sorted(
        [(cnt, s) for s, cnt in discovered_merges.items() if cnt >= 16],
        reverse=True,
    )[:25]
    if top_discovered:
        for _, s in top_discovered:
            tid = len(candidate_list)
            candidate_set.add(s)
            candidate_list.append(s)
            cand_t2i[s] = tid
            tot = sub_total[s]
            ed = sub_suffix[s] + sub_whole[s]
            ct = tot - ed
            bos_cost.append(-math.log2((sub_prefix[s] + sub_whole[s] + 1.0) / denom_start))
            uni_cost.append(-math.log2((sub_internal[s] + sub_suffix[s] + 1.0) / denom_mid))
            eos_cost.append(-math.log2((ed + 1.0) / (tot + 2.0)))
            cont_cost.append(-math.log2((ct + 1.0) / (tot + 2.0)))

        num_cands = len(candidate_list)
        for w_idx, w in enumerate(unique_words):
            n = len(w)
            spans_at_end = word_spans[w_idx]
            for i in range(1, n + 1):
                start_min = max(0, i - 6)
                existing_j = {j for j, _ in spans_at_end[i]}
                for j in range(start_min, i):
                    if j not in existing_j:
                        tid = cand_t2i.get(w[j:i])
                        if tid is not None:
                            spans_at_end[i].append((j, tid))

    active = [True] * num_cands
    BOS_IDX = num_cands
    smoothing_k = vocab_size + 1

    max_possible_count = total_chars + total_words + smoothing_k + 50
    LOG2_PLUS1 = [math.log2(c + 1.0) for c in range(max_possible_count)]
    LOG2_SMOOTH = [math.log2(r + smoothing_k) for r in range(max_possible_count)]
    DELTA_ROW = [
        (r + 1.0) * math.log2(r + smoothing_k) - r * math.log2(r + smoothing_k - 1.0)
        for r in range(max_possible_count)
    ]
    DELTA_PAIR = [0.0] + [
        (c + 1.0) * math.log2(c + 1.0) - c * math.log2(c)
        for c in range(1, max_possible_count)
    ]

    pair_counts = [[0] * (num_cands + 1) for _ in range(num_cands + 1)]
    row_counts = [0] * (num_cands + 1)

    for w_idx, seq in enumerate(word_segs):
        f = freqs[w_idx]
        prev = BOS_IDX
        for tid in seq:
            pair_counts[prev][tid] += f
            row_counts[prev] += f
            prev = tid
        pair_counts[prev][BOS_IDX] += f
        row_counts[prev] += f

    all_word_spans = [
        [list(spans) for spans in spans_at_end] for spans_at_end in word_spans
    ]

    def filter_active_spans():
        for w_idx, spans_at_end in enumerate(word_spans):
            orig = all_word_spans[w_idx]
            for i in range(1, len(spans_at_end)):
                spans_at_end[i] = [(j, tid) for j, tid in orig[i] if active[tid]]

    def run_online_loo_pass(alpha=0.40, token_bonus=0.0):
        one_m_a = 1.0 - alpha
        row_term = [
            one_m_a * LOG2_SMOOTH[r] + alpha * DELTA_ROW[r] - token_bonus
            for r in range(max_possible_count)
        ]
        row_term_eos = [rt + token_bonus for rt in row_term]
        pair_term = [
            one_m_a * LOG2_PLUS1[c] + alpha * DELTA_PAIR[c]
            for c in range(max_possible_count)
        ]

        for w_idx, spans_at_end in enumerate(word_spans):
            f = freqs[w_idx]
            old_seq = word_segs[w_idx]

            prev = BOS_IDX
            for tid in old_seq:
                pair_counts[prev][tid] -= f
                row_counts[prev] -= f
                prev = tid
            pair_counts[prev][BOS_IDX] -= f
            row_counts[prev] -= f

            n = len(spans_at_end) - 1
            dp = [[] for _ in range(n + 1)]
            bos_row_cost = row_term[row_counts[BOS_IDX]]
            bos_pairs = pair_counts[BOS_IDX]

            for i in range(1, n + 1):
                entries = []
                for j, tid in spans_at_end[i]:
                    if j == 0:
                        c = bos_row_cost - pair_term[bos_pairs[tid]]
                        entries.append((tid, c, 0, -1))
                    else:
                        best_c = 1e18
                        best_k = -1
                        for k_idx, (prev_tid, prev_c, _, _) in enumerate(dp[j]):
                            tc = (
                                prev_c
                                + row_term[row_counts[prev_tid]]
                                - pair_term[pair_counts[prev_tid][tid]]
                            )
                            if tc < best_c:
                                best_c = tc
                                best_k = k_idx
                        entries.append((tid, best_c, j, best_k))
                dp[i] = entries

            best_final_c = 1e18
            best_last_idx = 0
            for k_idx, (tid, c, _, _) in enumerate(dp[n]):
                fc = c + row_term_eos[row_counts[tid]] - pair_term[pair_counts[tid][BOS_IDX]]
                if fc < best_final_c:
                    best_final_c = fc
                    best_last_idx = k_idx

            new_seq = []
            curr_i = n
            curr_k = best_last_idx
            while curr_i > 0:
                tid, _, prev_j, prev_k = dp[curr_i][curr_k]
                new_seq.append(tid)
                curr_i = prev_j
                curr_k = prev_k
            new_seq.reverse()
            word_segs[w_idx] = new_seq

            prev = BOS_IDX
            for tid in new_seq:
                pair_counts[prev][tid] += f
                row_counts[prev] += f
                prev = tid
            pair_counts[prev][BOS_IDX] += f
            row_counts[prev] += f

    def get_fallback_decompositions(tid):
        s = candidate_list[tid]
        m = len(s)
        paths = []

        def dfs(pos, current_seq):
            if pos == m:
                if len(current_seq) >= 2:
                    internal_c = 0.0
                    for q in range(len(current_seq) - 1):
                        u, v = current_seq[q], current_seq[q + 1]
                        internal_c += LOG2_SMOOTH[row_counts[u]] - LOG2_PLUS1[pair_counts[u][v]]
                    uni_sum = sum(uni_cost[t] for t in current_seq)
                    paths.append((internal_c + 0.08 * uni_sum, current_seq[0], current_seq[-1], internal_c))
                return
            for nxt in range(pos + 1, m + 1):
                if pos == 0 and nxt == m:
                    continue
                sub_id = cand_t2i.get(s[pos:nxt])
                if sub_id is not None and active[sub_id]:
                    current_seq.append(sub_id)
                    dfs(nxt, current_seq)
                    current_seq.pop()

        dfs(0, [])
        paths.sort(key=lambda x: x[0])
        return [(u_f, u_l, inter_c) for _, u_f, u_l, inter_c in paths[:5]]

    def compute_delta_bits():
        fallbacks = [None] * num_cands
        for tid in range(num_chars, num_cands):
            if active[tid] and row_counts[tid] > 0:
                fallbacks[tid] = get_fallback_decompositions(tid)

        delta_bits = [0.0] * num_cands
        for w_idx, seq in enumerate(word_segs):
            f = freqs[w_idx]
            k_len = len(seq)
            for r, tid in enumerate(seq):
                if tid < num_chars:
                    continue
                left = BOS_IDX if r == 0 else seq[r - 1]
                right = BOS_IDX if r == k_len - 1 else seq[r + 1]

                r_left = max(0, row_counts[left] - f)
                c_lt = max(0, pair_counts[left][tid] - f)
                r_tid = max(0, row_counts[tid] - f)
                c_tr = max(0, pair_counts[tid][right] - f)
                c_with = (
                    (LOG2_SMOOTH[r_left] - LOG2_PLUS1[c_lt])
                    + (LOG2_SMOOTH[r_tid] - LOG2_PLUS1[c_tr])
                )

                best_without = 1e18
                for u_f, u_l, inter_c in fallbacks[tid]:
                    c_wo = (
                        (LOG2_SMOOTH[r_left] - LOG2_PLUS1[pair_counts[left][u_f]])
                        + inter_c
                        + (LOG2_SMOOTH[row_counts[u_l]] - LOG2_PLUS1[pair_counts[u_l][right]])
                    )
                    if c_wo < best_without:
                        best_without = c_wo

                delta_bits[tid] += f * (best_without - c_with)
        return delta_bits

    curr_active = sum(active)
    step_targets = [
        t for t in [345, 310, 286, 270, 260, vocab_size] if vocab_size <= t < curr_active
    ]
    if not step_targets or step_targets[-1] != vocab_size:
        step_targets.append(vocab_size)

    pruned_history = []

    for step_idx, target_sz in enumerate(step_targets):
        bonus = max(0.0, 0.14 - 0.025 * step_idx)
        alpha_val = 0.42 if step_idx < len(step_targets) - 1 else 0.35
        run_online_loo_pass(alpha=alpha_val, token_bonus=bonus)

        if sum(active) <= target_sz:
            continue

        delta_bits = compute_delta_bits()
        multi_cands = [
            (delta_bits[tid], row_counts[tid], tid)
            for tid in range(num_chars, num_cands)
            if active[tid]
        ]
        multi_cands.sort(reverse=True)
        keep_count = target_sz - num_chars
        keep_multi = set(tid for _, _, tid in multi_cands[:keep_count])

        for db, rc, tid in multi_cands[keep_count:]:
            if db > 0.0 and rc >= 8:
                pruned_history.append((db, rc, tid))

        for tid in range(num_chars, num_cands):
            if active[tid] and tid not in keep_multi:
                active[tid] = False

        filter_active_spans()

    if pruned_history:
        pruned_history.sort(reverse=True)
        revived = 0
        for _, _, tid in pruned_history:
            if not active[tid]:
                active[tid] = True
                revived += 1
                if revived >= 12:
                    break
        if revived > 0:
            filter_active_spans()
            run_online_loo_pass(alpha=0.35, token_bonus=0.04)
            run_online_loo_pass(alpha=0.30, token_bonus=0.0)
            delta_bits = compute_delta_bits()
            multi_cands = [
                (delta_bits[tid], row_counts[tid], tid)
                for tid in range(num_chars, num_cands)
                if active[tid]
            ]
            multi_cands.sort(reverse=True)
            keep_multi = set(tid for _, _, tid in multi_cands[: vocab_size - num_chars])
            for tid in range(num_chars, num_cands):
                if active[tid] and tid not in keep_multi:
                    active[tid] = False
            filter_active_spans()

    run_online_loo_pass(alpha=0.30, token_bonus=0.02)
    run_online_loo_pass(alpha=0.22, token_bonus=0.0)
    run_online_loo_pass(alpha=0.15, token_bonus=0.0)

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

    final_uni_tie = [
        -1e-4 * math.log2((final_row_counts[t] + 1.0) / (total_words + V))
        for t in range(V + 1)
    ]
    cost_matrix = [[0.0] * (V + 1) for _ in range(V + 1)]
    for u in range(V + 1):
        row_log2 = LOG2_SMOOTH[final_row_counts[u]]
        row_u = final_pair_counts[u]
        cost_u = cost_matrix[u]
        for v in range(V + 1):
            c = row_u[v]
            cost_u[v] = row_log2 - LOG2_PLUS1[c] + (final_uni_tie[v] if c == 0 else 0.0)

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