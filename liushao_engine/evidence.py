"""Dempster-Shafer fusion of heterogeneous detectors.

Two frames live here.

Legacy frame {attack, normal} (1.3.7, engine format 1; `bpa`, `combine`, `fuse`, unchanged):
each source score is mapped to a calibrated probability p (isotonic regression on
calibration data), then discounted by a reliability r (Shafer 1976):
m({attack}) = r p, m({normal}) = r (1 - p), m(Theta) = 1 - r.
Sources are combined with Dempster's rule; K is the mass of total conflict.

Open-world frame Theta = {A known attack, N normal, U unseen pattern} (1.3.8, T07; `fuse_open`).
The 8 subsets are bitmasks (A=1, N=2, U=4) and intersections come from a precomputed table.
Experts are modelled orthogonally (handoff 4.2):
  supervised (RF, LR)   attack -> {A};  not attack -> {N,U} (it never saw U and cannot exclude it)
  typicality            typical -> {N,A} (the seen world);  atypical -> {A,U} (not like normal)
  remainder of either   -> Theta
so {N,U} & {N,A} = {N}, {N,U} & {A,U} = {U} (novel) and {A} & anything = {A}.  Supervised and
typicality experts never have an empty intersection: K comes only from RF/LR disagreement.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .conformal import _finite_vector

EPS = 1e-6


CALIBRATORS = ('step', 'centred')


def fit_calibrator(score, y, kind='step'):
    """Isotonic calibrator stored as interpolation knots.

    kind='step' (1.3.7) keeps the block edges, so every score inside a block maps
    to the same value and the fused belief has long runs of ties.  kind='centred'
    (1.3.8, centred isotonic regression, Oron & Flournoy 2017) keeps one knot per
    block at the block's mean score and interpolates linearly between knots: the
    map stays monotone with the same block levels, but no longer collapses whole
    blocks onto one value.  Scores beyond the outermost knots are clipped.
    """
    if kind not in CALIBRATORS:
        raise ValueError('unknown calibrator kind: ' + str(kind))
    score, y = _finite_vector(score, 'score'), _finite_vector(y, 'y')
    # increasing='auto' lets calibration learn a detector's polarity (e.g. hyper-regular bot flows).
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing='auto', out_of_bounds='clip').fit(score, y)
    if kind == 'step':
        return {'x': iso.X_thresholds_.astype(np.float64), 'y': iso.y_thresholds_.astype(np.float64)}
    fitted = iso.predict(score)
    # Isotonic blocks are contiguous in score and have distinct levels, so the level identifies the block.
    levels, inverse = np.unique(fitted, return_inverse=True)
    centres = np.bincount(inverse, weights=score) / np.bincount(inverse)
    order = np.argsort(centres, kind='stable')
    return {'x': centres[order], 'y': levels[order], 'kind': 'centred'}


def calibrate(cal, score):
    score = _finite_vector(score, 'score')
    return np.clip(np.interp(score, cal['x'], cal['y']), EPS, 1 - EPS)


def reliability(score, y, cap=0.99):
    """Gini coefficient 2*AUC-1 as discount factor; a coin-flip source carries no mass."""
    return float(np.clip(2 * roc_auc_score(y, score) - 1, 0.0, cap))


def bpa(p, r):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([r * p, r * (1 - p), np.full_like(p, 1 - r)], axis=1)


def combine(masses):
    """Dempster's rule over any number of BPAs (columns: attack, normal, theta).

    Returns normalised (attack, normal, theta) and the total conflict K, i.e. the
    unnormalised conjunctive mass on the empty set.
    """
    a, n, u = masses[0][:, 0], masses[0][:, 1], masses[0][:, 2]
    empty = np.zeros_like(a)
    for m in masses[1:]:
        a2, n2, u2 = m[:, 0], m[:, 1], m[:, 2]
        empty = empty + a * n2 + n * a2
        a, n, u = a * a2 + a * u2 + u * a2, n * n2 + n * u2 + u * n2, u * u2
    total = np.maximum(1 - empty, EPS)
    return a / total, n / total, u / total, empty


def validated_scores(scores, sources):
    """One finite vector per requested source, aligned before interpolation or tail lookup."""
    values = {name: _finite_vector(scores[name], name + ' score') for name in sources}
    if not values:
        raise ValueError('至少需要一个证据源')
    if len({len(value) for value in values.values()}) != 1:
        raise ValueError('各证据源的分数必须等长')
    return values


def fuse(scores, params):
    scores = validated_scores(scores, params['sources'])
    masses = [bpa(calibrate(params['calibrators'][name], scores[name]), params['reliability'][name])
              for name in params['sources']]
    belief, _, ignorance, conflict = combine(masses)
    return {'ds_belief': belief, 'ds_plausibility': belief + ignorance,
            'ds_conflict': conflict, 'ds_betp': belief + ignorance / 2}


# ----------------------------------------------------------------------------- open-world frame (1.3.8, T07)
A, N, U = 1, 2, 4
THETA = A | N | U
SUBSETS = 8
NAMES = ('∅', 'A', 'N', 'A,N', 'U', 'A,U', 'N,U', 'A,N,U')
INTERSECTION = np.array([[i & j for j in range(SUBSETS)] for i in range(SUBSETS)])
CARDINALITY = np.array([bin(i).count('1') for i in range(SUBSETS)])
# For every target subset, the (i, j) pairs of non-empty focal sets whose intersection is the target,
# in ascending (i, j) order.  The order is part of the contract: on masses confined to {A}, {N}, {A,N}
# the sums below add the same terms in the same order as the legacy `combine`, so the results are
# bit-identical (the extra terms are exact zeros).
PAIRS = tuple(tuple((i, j) for i in range(1, SUBSETS) for j in range(1, SUBSETS) if INTERSECTION[i, j] == t)
              for t in range(SUBSETS))
KINDS = ('supervised', 'typicality')
ATYPICAL_FOCAL = {'AU': A | U, 'U': U}


def masses(assignments, rows):
    """(rows, 8) mass array from {subset bitmask: per-record mass}; column 0 (the empty set) stays 0."""
    out = np.zeros((rows, SUBSETS))
    for subset, value in assignments.items():
        if not 0 < subset < SUBSETS:
            raise ValueError('焦元必须是非空子集的位掩码')
        out[:, subset] = out[:, subset] + value
    return out


def supervised_masses(p, r, theta=THETA):
    """Attack detector: r p on {A}, r (1 - p) on "not a known attack" (theta without A), 1 - r on theta."""
    p = np.asarray(p, dtype=np.float64)
    return masses({A: r * p, theta & ~A: r * (1 - p), theta: np.full_like(p, 1 - r)}, len(p))


def typicality_masses(q, r, theta=THETA, atypical='AU'):
    """Typicality expert with atypical degree q: r q on "not like normal", r (1 - q) on the seen world {N,A}."""
    q = np.asarray(q, dtype=np.float64)
    return masses({ATYPICAL_FOCAL[atypical] & theta: r * q, (N | A) & theta: r * (1 - q),
                   theta: np.full_like(q, 1 - r)}, len(q))


def combine_masses(sources):
    """Dempster's rule on the 8-subset frame.

    sources: non-empty list of (n, 8) arrays with no mass on the empty set.  Returns (normalised
    masses (n, 8) with column 0 zero, conflict K).  As in the legacy `combine`, conflict is the
    unnormalised conjunctive mass on the empty set accumulated over the sources and the result is
    normalised once; under total conflict (K = 1) every normalised mass is 0.
    """
    if not len(sources):
        raise ValueError('至少需要一个证据源')
    m = np.asarray(sources[0], dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != SUBSETS:
        raise ValueError('质量数组必须是 (n, 8)')
    empty = np.zeros(len(m))
    for other in sources[1:]:
        other = np.asarray(other, dtype=np.float64)
        if other.shape != m.shape:
            raise ValueError('各证据源的质量数组形状不一致')
        for i, j in PAIRS[0]:
            empty = empty + m[:, i] * other[:, j]
        new = np.zeros_like(m)
        for target in range(1, SUBSETS):
            for i, j in PAIRS[target]:
                new[:, target] = new[:, target] + m[:, i] * other[:, j]
        m = new
    total = np.maximum(1 - empty, EPS)
    out = np.zeros_like(m)
    for target in range(1, SUBSETS):
        out[:, target] = m[:, target] / total
    return out, empty


def belief(m, subset):
    """Bel(subset): mass of the non-empty subsets contained in it."""
    total = np.zeros(len(m))
    for s in range(1, SUBSETS):
        if s & ~subset == 0:
            total = total + m[:, s]
    return total


def plausibility(m, subset):
    """Pl(subset): mass of the subsets that intersect it (ascending bitmask order)."""
    total = np.zeros(len(m))
    for s in range(1, SUBSETS):
        if s & subset:
            total = total + m[:, s]
    return total


def pignistic(m, element):
    """BetP(element) = sum over subsets containing it of m(S) / |S| (Smets & Kennes 1994)."""
    total = np.zeros(len(m))
    for s in range(1, SUBSETS):
        if s & element:
            total = total + (m[:, s] if CARDINALITY[s] == 1 else m[:, s] / CARDINALITY[s])
    return total


def fit_atypical_map(null_scores, alpha=0.05):
    """Tail map of the typicality expert, fixed on reference normals (validation half A)."""
    null = np.sort(_finite_vector(null_scores, 'null_scores'))
    if not len(null) or not np.isfinite(null).all() or not 0 < alpha < 1:
        raise ValueError('典型性零分布需要有限的分数和 (0, 1) 内的 alpha')
    return {'null': null, 'alpha': float(alpha)}


def atypical_degree(tail_map, score):
    """q = clip(1 - p / alpha, 0, 1) with p = (1 + #{null >= s}) / (n + 1); 0 up to the null's (1 - alpha) tail."""
    null = tail_map['null']
    s = _finite_vector(score, 'score')
    p = (1 + len(null) - np.searchsorted(null, s, side='left')) / (len(null) + 1)
    return np.clip(1 - p / tail_map['alpha'], 0.0, 1.0)


def source_mass(name, score, params):
    """(n, 8) masses of one source under fusion parameters `params` (frame, kinds, calibrators, ...)."""
    theta, kind, r = params['theta'], params['kinds'][name], params['reliability'][name]
    if kind == 'supervised':
        return supervised_masses(calibrate(params['calibrators'][name], score), r, theta)
    if kind == 'typicality':
        return typicality_masses(atypical_degree(params['atypical_map'], score), r, theta, params['atypical_focal'])
    raise ValueError('unknown source kind: ' + str(kind))


def novelty_share(bel_unknown, bel_normal):
    """Bel(U) / (Bel(U) + Bel(N)), 0 where both vanish: how the not-a-known-attack mass leans to "unseen".

    With the orthogonal focal sets this is the typicality expert's atypical degree q and does not
    depend on the supervised experts.
    """
    u, n = np.asarray(bel_unknown, dtype=np.float64), np.asarray(bel_normal, dtype=np.float64)
    total = u + n
    return np.divide(u, total, out=np.zeros_like(total), where=total > 0)


def fuse_open(scores, params):
    """Orthogonal evidence fusion; `params['theta']` is the frame (A|N|U, or A|N for the degenerate frame).

    Returns the legacy columns for the attack hypothesis (same names) plus the normal / unseen ones.
    """
    scores = validated_scores(scores, params['sources'])
    m, conflict = combine_masses([source_mass(name, scores[name], params) for name in params['sources']])
    return {'ds_belief': belief(m, A), 'ds_plausibility': plausibility(m, A), 'ds_conflict': conflict,
            'ds_betp': pignistic(m, A), 'ds_bel_normal': belief(m, N), 'ds_bel_unknown': belief(m, U),
            'ds_betp_normal': pignistic(m, N), 'ds_betp_unknown': pignistic(m, U)}
