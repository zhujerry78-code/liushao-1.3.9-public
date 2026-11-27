"""Typicality expert candidates for 1.3.8 (T06): "does this record look like the world we have seen?".

Everything here is unsupervised statistics on reference data plus read-only use of the frozen RF
structure; nothing retrains the base model or changes its threshold.

A typicality model is fitted per reference class (normal / known attack / mixed) and optionally per
traffic bucket, and turned into a conformal p-value on held-out records of the same class.  The
atypicality score is -log p; for the split reference it is -log max(p_normal, p_attack), i.e. a record
is typical when it resembles either normal traffic or a known attack.

Spaces:
  raw     per-feature ECOD tails on the 77 features
  pca     signed log1p + robust scaling, PCA fitted on the normal reference, ECOD tails on all components
  rfnode  frozen RF trees cut at depth d; reference count in the node a record reaches, log-averaged
Aggregation for raw/pca: 'sum' (ECOD's max of left/right/auto sums) or ('topk', k) (mean of the k largest
per-dimension tails max(left, right)).
"""
import numpy as np
import pandas as pd

from .ecod import _feature_matrix, _sum_tail_logs, fit_ecod, tail_logs
from .conformal import _finite_vector

PORT_BUCKETS = {53: 'p53', 80: 'p80', 443: 'p443', 22: 'p22', 21: 'p21', 3389: 'p3389', 445: 'p445'}
GLOBAL = '__global__'
MISSING = '__missing__'
OTHER = 'other'
MIN_FIT_ROWS = 1000
MIN_CAL_ROWS = 500


# ----------------------------------------------------------------------------- buckets
def bucket_keys(kind, features, meta=None):
    """Bucket label per record.  'dst_port' falls back to MISSING (-> global model) when the column is absent."""
    n = len(features)
    if kind == 'none':
        return np.full(n, GLOBAL, dtype=object)
    if kind == 'protocol':
        proto = pd.to_numeric(pd.Series(np.asarray(features['Protocol'])), errors='coerce')
        return np.where(proto.isna(), OTHER, 'proto' + proto.fillna(-1).astype(int).astype(str)).astype(object)
    if kind == 'dst_port':
        if meta is None or 'Dst Port' not in meta:
            return np.full(n, MISSING, dtype=object)
        port = pd.to_numeric(pd.Series(np.asarray(meta['Dst Port'])), errors='coerce').to_numpy()
        keys = np.where(np.isnan(port), OTHER, np.where(port < 1024, 'wellknown', 'high')).astype(object)
        for p, name in PORT_BUCKETS.items():
            keys[port == p] = name
        return keys
    raise ValueError('unknown bucketing: ' + str(kind))


def merged_buckets(fit_keys, cal_keys):
    """Buckets with enough fit and calibration rows keep their own model; the rest pool into OTHER."""
    fit_counts, cal_counts = pd.Series(fit_keys).value_counts(), pd.Series(cal_keys).value_counts()
    keep = {k for k in fit_counts.index if k not in (GLOBAL, MISSING, OTHER)
            and fit_counts[k] >= MIN_FIT_ROWS and cal_counts.get(k, 0) >= MIN_CAL_ROWS}
    return keep


def _route(keys, keep):
    keys = np.asarray(keys, dtype=object)
    return np.where(np.isin(keys, list(keep)), keys, OTHER).astype(object)


# ----------------------------------------------------------------------------- spaces
def fit_pca(x_normal):
    x = np.asarray(x_normal, dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    z = np.sign(np.where(np.isfinite(x), x, med)) * np.log1p(np.abs(np.where(np.isfinite(x), x, med)))
    centre = np.median(z, axis=0)
    q75, q25 = np.percentile(z, [75, 25], axis=0)
    scale = np.where(q75 - q25 > 0, q75 - q25, np.where(z.std(axis=0) > 0, z.std(axis=0), 1.0))
    z = (z - centre) / scale
    mean = z.mean(axis=0)
    _, _, vt = np.linalg.svd(z - mean, full_matrices=False)
    return {'median': med, 'centre': centre, 'scale': scale, 'mean': mean, 'components': vt}


def pca_transform(model, x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, model['median'])
    z = (np.sign(x) * np.log1p(np.abs(x)) - model['centre']) / model['scale']
    return (z - model['mean']) @ model['components'].T


def tree_ancestors(tree, depth):
    """For every node, its ancestor at the given depth (itself when the node is shallower)."""
    left, right = tree.children_left, tree.children_right
    n = tree.node_count
    parent = np.full(n, -1)
    for child in (left, right):
        has = child != -1
        parent[child[has]] = np.flatnonzero(has)
    node_depth = np.zeros(n, dtype=int)
    for _ in range(int(tree.max_depth)):  # depth = number of steps to the root
        node_depth = np.where(parent >= 0, node_depth[np.maximum(parent, 0)] + 1, 0)
    anc = np.arange(n)
    for _ in range(int(tree.max_depth)):
        anc = np.where(node_depth[anc] > depth, parent[anc], anc)
    return anc


def rf_leaves(rf_pipeline, x):
    """(n, trees) leaf ids of the frozen RF, through its own preprocessing; read-only."""
    pre, clf = rf_pipeline[:-1], rf_pipeline[-1]
    return clf.apply(pre.transform(x)).astype(np.int32)


def rf_cut(rf_pipeline, depth):
    return [tree_ancestors(est.tree_, depth) for est in rf_pipeline[-1].estimators_]


def fit_rfnode(leaves, cut):
    nodes = np.column_stack([cut[t][leaves[:, t]] for t in range(leaves.shape[1])])
    return {'counts': [np.bincount(nodes[:, t], minlength=len(cut[t])) for t in range(nodes.shape[1])],
            'n': int(len(leaves))}


def score_rfnode(model, leaves, cut):
    n = model['n']
    logs = np.column_stack([np.log((model['counts'][t][cut[t][leaves[:, t]]] + 1) / (n + 1))
                            for t in range(leaves.shape[1])])
    return -logs.mean(axis=1)


# ----------------------------------------------------------------------------- ECOD-style aggregation
def aggregate(left, right, skew_sign, how):
    if how == 'sum':
        return _sum_tail_logs(left, right, skew_sign)
    if not isinstance(how, (tuple, list)) or len(how) != 2:
        raise ValueError('aggregation must be sum or (topk, positive integer)')
    kind, k = how
    if (kind != 'topk' or isinstance(k, (bool, np.bool_))
            or not isinstance(k, (int, np.integer)) or k < 1):
        raise ValueError('topk must use a positive integer')
    tail = np.maximum(left, right)
    return -np.sort(-tail, axis=1)[:, :k].mean(axis=1)


def all_aggregations(tail_model, z, hows):
    left, right = tail_logs(tail_model, z)
    return {how: aggregate(left, right, tail_model['skew_sign'], how) for how in hows}


# ----------------------------------------------------------------------------- conformal p-values
def p_values(cal_scores, scores):
    """p = (1 + #{cal >= s}) / (n_cal + 1); larger score = more atypical."""
    cal = np.sort(_finite_vector(cal_scores, 'cal_scores'))
    s = _finite_vector(scores, 'scores')
    return (1 + len(cal) - np.searchsorted(cal, s, side='left')) / (len(cal) + 1)


def bucketed_p(fit_keys, cal_keys, query_keys, fit_fn, score_fn):
    """Fit one model per kept bucket (+ OTHER + GLOBAL), calibrate on held-out rows, return query p-values.

    fit_fn(mask_fit) -> model; score_fn(model, which, mask) -> {variant: scores} for which in ('cal', q).
    Query records whose bucket has no model (unseen key, MISSING column) use the GLOBAL model.
    Returns ({variant: p-values per query set}, info).
    """
    keep = merged_buckets(fit_keys, cal_keys)
    fit_r, cal_r = _route(fit_keys, keep), _route(cal_keys, keep)
    plan = [(b, fit_r == b, cal_r == b) for b in sorted(keep)]
    other_ok = (fit_r == OTHER).sum() >= MIN_FIT_ROWS and (cal_r == OTHER).sum() >= MIN_CAL_ROWS
    if other_ok and keep:
        plan.append((OTHER, fit_r == OTHER, cal_r == OTHER))
    plan.append((GLOBAL, np.ones(len(fit_r), bool), np.ones(len(cal_r), bool)))
    has_model = {b for b, _, _ in plan}
    routed = {name: np.where(np.isin(_route(k, keep), list(has_model)), _route(k, keep), GLOBAL)
              for name, k in query_keys.items()}
    out, info = {}, {'buckets': {}}
    for b, fm, cm in plan:
        model = fit_fn(fm)
        cal = score_fn(model, 'cal', cm)
        for name, r in routed.items():
            qm = r == b
            if b == GLOBAL:
                qm = r == GLOBAL
            if not qm.any():
                continue
            q = score_fn(model, name, qm)
            for variant, s in q.items():
                arr = out.setdefault(variant, {}).setdefault(name, np.full(len(r), np.nan))
                arr[qm] = p_values(cal[variant], s)
        info['buckets'][b] = {'fit_rows': int(fm.sum()), 'cal_rows': int(cm.sum()),
                              'query_rows': {n: int((r == b).sum()) for n, r in routed.items()}}
    return out, info


def split_reference(rows, max_rows=20000, seed=42):
    """Fit/calibration split of one reference class (same draw as fit_ecod's subsample)."""
    fit = np.zeros(rows, dtype=bool)
    if rows <= max_rows:
        raise ValueError('参考类样本不足以留出校准集')
    fit[np.random.default_rng(seed).choice(rows, max_rows, replace=False)] = True
    return fit


def fit_tail_model(z):
    z = _feature_matrix(z, 'z', require_rows=True)
    return fit_ecod(z, max_rows=len(z) + 1)


# ----------------------------------------------------------------------------- the T06 pick, as one model
SELECTED = 'raw|normal_only|none|top3'  # configs/t06_selection_rules.json, result t06_grid_20261111


def fit_typicality(x_train_normal, how=('topk', 3), max_rows=20000, seed=42):
    """The selected signal: ECOD tails on train normals, top-k mean, conformal p on the held-out normals."""
    x = _feature_matrix(x_train_normal, 'x_train_normal', require_rows=True)
    fit = split_reference(len(x), max_rows, seed)
    tail = fit_tail_model(x[fit])
    left, right = tail_logs(tail, x[~fit])
    return {'config': SELECTED, 'tail': tail, 'how': how,
            'calibration': np.sort(aggregate(left, right, tail['skew_sign'], how))}


def score_typicality(model, x):
    """Atypicality -log p (larger = less like the reference normals)."""
    left, right = tail_logs(model['tail'], x)
    return _score_typicality_from_tails(model, left, right, model['tail']['skew_sign'])


def _score_typicality_from_tails(model, left, right, skew_sign):
    return -np.log(p_values(model['calibration'], aggregate(left, right, skew_sign, model['how'])))
