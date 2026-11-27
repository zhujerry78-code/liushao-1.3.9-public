"""Output-space drift detection with a time-block null (1.3.8, T08).

A batch is compared with a labelled reference (validation half A) in the space of the three expert
outputs z = (RF score, LR score, atypicality).  Four candidate statistics, all in [0, 1]
(configs/t08_drift_rules.json; the benchmark script picks one by that pre-registered rule):

  whole_mmd       saturation index MMD^2 / (mean k_xx + mean k_yy) of the unbiased RBF-kernel MMD^2
                  (Gretton et al. 2012) on standardised z; per expert = the same on one coordinate.
  tail_h          per expert and tail level q: share of the batch above the reference-normal q-quantile
                  vs. the reference share, as Cohen's h = 2 asin sqrt(p) - 2 asin sqrt(p0);
                  delta_s = max_q |h| / pi, delta = max_s delta_s.
  whole_mmd_adj,  the same two, against a reference re-mixed to the batch's estimated share of known
  tail_h_adj      attacks w_hat (black-box shift estimation, Lipton et al. 2018, binary closed form with
                  the frozen RF decision), so a mere change in how many known attacks there are is
                  not counted as drift.

Null ("same environment, another time period"): every reference time block in turn is the pseudo-batch
and the remaining blocks are the reference; pseudo-batches are redrawn at the batch-size classes in
SIZES.  A batch drifts when its statistic exceeds the 95% quantile of the pooled null of its size class.
Nothing here touches the frozen base model; inputs are scores only.
"""
import numpy as np

EXPERTS = ('rf', 'lr', 'typ')
NAMES = ('joint',) + EXPERTS
CANDIDATES = ('whole_mmd', 'tail_h', 'whole_mmd_adj', 'tail_h_adj')
SELECTED = 'tail_h_adj'  # configs/t08_drift_rules.json, result t08_drift_20261111, owner decision 2026-11-11
SIZES = (200, 500, 1000)
TAIL_LEVELS = (0.90, 0.95, 0.99)
MMD_ROWS = 1000
MIN_BLOCK_ROWS = 200
NULL_QUANTILE = 0.95
SEED = 138
RF = 0  # coordinate of the frozen RF score


def _real_array(value, name):
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError(name + '不得包含被掩码隐藏的缺失值')
    array = np.asarray(value)
    if array.dtype.kind not in 'biuf':
        raise ValueError(name + '必须为实数，不接受复数或非数值输入')
    return array


def _score_matrix(z):
    z = _real_array(z, '漂移专家分数')
    if z.ndim != 2 or z.shape[1] != len(EXPERTS):
        raise ValueError('漂移检测需要有限的 (n, 3) 专家分数')
    z = z.astype(np.float64, copy=False)
    if not np.isfinite(z).all():
        raise ValueError('漂移检测需要有限的 (n, 3) 专家分数')
    return z


def _label_vector(y, rows):
    y = _real_array(y, '漂移参考标签')
    if y.ndim != 1 or len(y) != rows or not np.isin(y, [0, 1]).all():
        raise ValueError('漂移参考标签须为与专家分数等长的一维 0/1 数组')
    return y.astype(int, copy=False)


def _rf_threshold(value):
    value = _real_array(value, '漂移 RF 阈值')
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError('漂移 RF 阈值必须为有限实数标量')
    return float(value)


def _block_vector(block_ids):
    if np.ma.isMaskedArray(block_ids) and np.ma.getmaskarray(block_ids).any():
        raise ValueError('漂移时间块不得包含缺失值')
    # Preserve missing scalars in mixed string/numeric lists until validation.
    ids = np.asarray(block_ids, dtype=object)
    if ids.ndim != 1:
        raise ValueError('漂移时间块必须为一维数组')
    for value in ids:
        try:
            missing = value is None or np.ma.is_masked(value) or bool(value != value)
        except (TypeError, ValueError):
            raise ValueError('漂移时间块不得包含缺失或非标量标识') from None
        if missing:
            raise ValueError('漂移时间块不得包含缺失值')
    return np.asarray(block_ids)


# ----------------------------------------------------------------------------- space and reference
def _sq1(a, b):
    d = a[:, None] - b[None, :]
    return d * d


def median_bandwidth(z, max_rows=2000, seed=SEED):
    """Median heuristic on (a subsample of) z; z is (n, d)."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) > max_rows:
        z = z[np.random.default_rng(seed).choice(len(z), max_rows, replace=False)]
    d = sum(_sq1(z[:, j], z[:, j]) for j in range(z.shape[1]))[np.triu_indices(len(z), k=1)]
    d = d[d > 0]
    return float(np.sqrt(np.median(d) / 2)) if len(d) else 1.0


def fit_space(z, seed=SEED):
    """Standardisation and kernel bandwidths, fixed on the reference."""
    z = np.asarray(z, dtype=np.float64)
    mu, sd = z.mean(axis=0), z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    zs = (z - mu) / sd
    bandwidth = {'joint': median_bandwidth(zs, seed=seed)}
    bandwidth.update({e: median_bandwidth(zs[:, [j]], seed=seed) for j, e in enumerate(EXPERTS)})
    return {'mu': mu, 'sd': sd, 'bandwidth': bandwidth}


def reference_stats(z, y, rf_threshold):
    """Class-wise summaries of a labelled reference (y: 0 normal, 1 known attack)."""
    z = _score_matrix(z)
    y = _label_vector(y, len(z))
    rf_threshold = _rf_threshold(rf_threshold)
    normal, attack = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(normal):
        raise ValueError('漂移参考集没有正常流量')
    levels = np.asarray(TAIL_LEVELS)
    cuts = np.quantile(z[normal], levels, axis=0).T  # (experts, levels)

    def above(rows):
        return (z[rows][:, :, None] > cuts[None]).mean(axis=0) if len(rows) else np.zeros_like(cuts)
    hit = z[:, RF] >= rf_threshold
    tpr = float(hit[attack].mean()) if len(attack) else np.nan
    return {'z': z, 'normal': normal, 'attack': attack, 'cuts': cuts,
            'above_all': above(np.arange(len(z))), 'above_normal': above(normal), 'above_attack': above(attack),
            'pi': float(len(attack) / len(z)), 'tpr': tpr, 'fpr': float(hit[normal].mean()),
            'rf_threshold': float(rf_threshold)}


def attack_share(ref, z):
    """w_hat: the batch's share of known attacks, from the frozen RF decision corrected by the reference confusion."""
    if not len(ref['attack']) or not ref['tpr'] > ref['fpr']:
        return ref['pi']
    q = float((np.asarray(z)[:, RF] >= ref['rf_threshold']).mean())
    return float(np.clip((q - ref['fpr']) / (ref['tpr'] - ref['fpr']), 0.0, 1.0))


# ----------------------------------------------------------------------------- statistics
def _index(dxx, dyy, dxy, bandwidth):
    g = 1.0 / (2 * bandwidth ** 2)
    n, m = len(dxx), len(dyy)
    a = (np.exp(-g * dxx).sum() - n) / (n * (n - 1))  # diagonal of k is 1
    b = (np.exp(-g * dyy).sum() - m) / (m * (m - 1))
    c = np.exp(-g * dxy).mean()
    return float(max((a + b - 2 * c) / (a + b), 0.0)) if a + b > 0 else 0.0


def mmd_statistic(space, ref, z, rng, adjusted, w=None):
    """Saturation indices (joint + per expert) of z against a subsample of the reference."""
    rows = min(MMD_ROWS, len(ref['z']))
    if adjusted:
        k = min(int(round(rows * w)), len(ref['attack']))
        pick = np.r_[rng.choice(ref['attack'], k, replace=False),
                     rng.choice(ref['normal'], min(rows - k, len(ref['normal'])), replace=False)]
    else:
        pick = rng.choice(len(ref['z']), rows, replace=False)
    x = (np.asarray(z, dtype=np.float64) - space['mu']) / space['sd']
    r = (ref['z'][pick] - space['mu']) / space['sd']
    out, total = {}, [0.0, 0.0, 0.0]
    for j, e in enumerate(EXPERTS):
        d = (_sq1(x[:, j], x[:, j]), _sq1(r[:, j], r[:, j]), _sq1(x[:, j], r[:, j]))
        out[e] = _index(*d, space['bandwidth'][e])
        total = [t + v for t, v in zip(total, d)]
    out['joint'] = _index(*total, space['bandwidth']['joint'])
    return out


def tail_statistic(ref, z, adjusted, w=None):
    """|Cohen's h| / pi of tail exceedance shares; per expert the maximum over tail levels."""
    p = (np.asarray(z, dtype=np.float64)[:, :, None] > ref['cuts'][None]).mean(axis=0)
    p0 = (1 - w) * ref['above_normal'] + w * ref['above_attack'] if adjusted else ref['above_all']
    h = np.abs(2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(p0))) / np.pi
    out = {e: float(h[j].max()) for j, e in enumerate(EXPERTS)}
    out['joint'] = max(out.values())
    return out


def statistic(space, ref, z, candidate, rng):
    """{'joint', 'rf', 'lr', 'typ', 'w_hat'} for one batch (already cut to its size class where needed)."""
    if candidate not in CANDIDATES:
        raise ValueError('unknown drift statistic: ' + str(candidate))
    adjusted = candidate.endswith('_adj')
    w = attack_share(ref, z) if adjusted else None
    out = (mmd_statistic(space, ref, z, rng, adjusted, w) if candidate.startswith('whole_mmd')
           else tail_statistic(ref, z, adjusted, w))
    out['w_hat'] = w
    return out


# ----------------------------------------------------------------------------- null distribution
def _rng(*key):
    return np.random.default_rng([int(k) for k in key])


def null_unit(space, z, y, batch_rows, ref_rows, rf_threshold, candidates, sizes, reps, key):
    """Null draws for one pseudo-batch pool (a time block) against one reference (the other blocks).

    Returns {candidate: {size: array (reps, len(NAMES))}}.  Draws are keyed by (key, size, rep), so the
    result does not depend on execution order or on which other candidates are computed.
    """
    ref = reference_stats(z[ref_rows], y[ref_rows], rf_threshold)
    out = {c: {m: np.empty((reps, len(NAMES))) for m in sizes} for c in candidates}
    for m in sizes:
        for r in range(reps):
            rows = _rng(*key, m, r).choice(batch_rows, min(m, len(batch_rows)), replace=False)
            for c in candidates:
                s = statistic(space, ref, z[rows], c, _rng(*key, m, r, 1 + CANDIDATES.index(c)))
                out[c][m][r] = [s[n] for n in NAMES]
    return out


def usable_blocks(block_ids, min_rows=MIN_BLOCK_ROWS):
    """({block: row indices} for blocks with enough rows, {dropped block: rows})."""
    ids = _block_vector(block_ids).astype(str)
    uniq, inv, counts = np.unique(ids, return_inverse=True, return_counts=True)
    keep = {str(b): np.flatnonzero(inv == i) for i, b in enumerate(uniq) if counts[i] >= min_rows}
    return keep, {str(b): int(c) for b, c in zip(uniq, counts) if c < min_rows}


def _parallel(jobs, n_jobs):
    if n_jobs == 1:
        return [f(*a) for f, a in jobs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(f)(*a) for f, a in jobs)


def pooled_thresholds(draws, sizes, quantile=NULL_QUANTILE):
    """draws: list over units of {size: (reps, names)} -> ({size: {name: q}}, summary)."""
    thresholds, summary = {}, {}
    for m in sizes:
        pooled = np.vstack([d[m] for d in draws])
        thresholds[m] = {n: float(np.quantile(pooled[:, i], quantile)) for i, n in enumerate(NAMES)}
        summary[m] = {'units': len(draws), 'draws': int(len(pooled)),
                      'joint_median': float(np.median(pooled[:, 0])), 'joint_max': float(pooled[:, 0].max()),
                      'joint_unit_medians': [float(np.median(d[m][:, 0])) for d in draws]}
    return thresholds, summary


def fit_drift(z, y, block_ids, *, rf_threshold, candidates=CANDIDATES, sizes=SIZES, reps=200, seed=SEED,
              min_block_rows=MIN_BLOCK_ROWS, n_jobs=1, keep_null=False):
    """Fit the reference, the comparison space and the leave-one-block-out null thresholds."""
    z = _score_matrix(z)
    y = _label_vector(y, len(z))
    rf_threshold = _rf_threshold(rf_threshold)
    block_ids = _block_vector(block_ids)
    if len(block_ids) != len(z):
        raise ValueError('漂移时间块须与专家分数等长')
    blocks, dropped = usable_blocks(block_ids, min_block_rows)
    if len(blocks) < 2:
        raise ValueError('漂移零分布至少需要两个时间块')
    space = fit_space(z, seed)
    everything = np.arange(len(z))
    names = sorted(blocks)
    jobs = [(null_unit, (space, z, y, blocks[b], np.setdiff1d(everything, blocks[b]), rf_threshold,
                         tuple(candidates), tuple(sizes), reps, (seed, i))) for i, b in enumerate(names)]
    units = _parallel(jobs, n_jobs)
    model = {'kind': 'output-space-drift', 'experts': list(EXPERTS), 'candidates': list(candidates),
             'candidate': SELECTED if SELECTED in candidates else None, 'sizes': list(sizes), 'tail_levels': list(TAIL_LEVELS), 'quantile': NULL_QUANTILE,
             'seed': int(seed), 'reps': int(reps), 'rf_threshold': float(rf_threshold), 'space': space,
             'reference': reference_stats(z, y, rf_threshold), 'reference_labels': y,
             'null_form': 'leave-one-block-out', 'blocks': {b: int(len(blocks[b])) for b in names},
             'dropped_blocks': dropped, 'thresholds': {}, 'null_summary': {}}
    for c in candidates:
        model['thresholds'][c], model['null_summary'][c] = pooled_thresholds([u[c] for u in units], sizes)
    if keep_null:
        model['null'] = {c: {b: u[c] for b, u in zip(names, units)} for c in candidates}
    return model


# ----------------------------------------------------------------------------- detection
def size_class(model, rows):
    fit = [m for m in model['sizes'] if m <= rows]
    return max(fit) if fit else None


def detect_drift(model, z, candidate=None, seed=None):
    """Batch-level drift verdict: delta, per-expert delta_s, thresholds of the batch's size class."""
    candidate = candidate or model['candidate']
    if candidate not in model['thresholds']:
        raise ValueError('漂移模型没有该统计量的阈值: ' + str(candidate))
    z = _score_matrix(z)
    m = size_class(model, len(z))
    if m is None:
        return {'status': 'too_small', 'candidate': candidate, 'rows': int(len(z)), 'drift': None}
    rng = np.random.default_rng(model['seed'] if seed is None else seed)
    used = z[rng.choice(len(z), m, replace=False)] if candidate.startswith('whole_mmd') and len(z) > m else z
    s = statistic(model['space'], model['reference'], used, candidate, rng)
    t = model['thresholds'][candidate][m]
    return {'status': 'ok', 'candidate': candidate, 'rows': int(len(z)), 'rows_used': int(len(used)), 'size_class': m,
            'delta': s['joint'], 'threshold': t['joint'], 'drift': bool(s['joint'] > t['joint']),
            'delta_s': {e: s[e] for e in EXPERTS}, 'threshold_s': {e: t[e] for e in EXPERTS},
            'drift_s': {e: bool(s[e] > t[e]) for e in EXPERTS}, 'w_hat': s['w_hat']}
