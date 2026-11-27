"""Label-conditional (Mondrian) split conformal prediction (Vovk et al. 2005).

Nonconformity of label "attack" is 1 - BetP(attack).  For label "normal" it is BetP(attack) on the
legacy two-element frame and 1 - BetP(normal) on the open-world frame (1.3.8, T07,
`scores='class-specific-betp'`); the two coincide when there is no "unseen" hypothesis.
p_y(x) = (#{calibration of class y with alpha_i >= alpha_y(x)} + 1) / (n_y + 1),
so for each class P(y in set) >= 1 - epsilon under exchangeability.  On the open-world frame a
record whose mass sits on "unseen" conforms to neither class and gets the empty set.

`scores='orthogonal-tests'` (T07b) tests the normal class twice, once per orthogonal expert group:
BetP(attack) (supervised evidence) and the novelty share Bel(U) / (Bel(U) + Bel(N)) (typicality
evidence), each calibrated on the normal calibration records, combined by Bonferroni:
p_normal = min(1, 2 min(p_supervised, p_typicality)).  Normal coverage >= 1 - epsilon still holds
(union bound) and the two kinds of evidence no longer compete for one rejection budget.
"""
import numpy as np

NORMAL, ATTACK, UNSURE = '正常', '攻击', '不确定'
CLASS_SPECIFIC = 'class-specific-betp'
ORTHOGONAL = 'orthogonal-tests'


def fit_conformal(betp, y, epsilon=0.05, betp_normal=None, novelty_share=None):
    betp, y = _finite_vector(betp, 'betp'), _finite_vector(y, 'y')
    epsilon = _epsilon(epsilon)
    if len(betp) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('校准分数和标签必须等长，且标签只能为 0 或 1')
    if novelty_share is not None:
        share = _aligned_vector(novelty_share, 'novelty_share', len(betp))
        return {'attack': np.sort(1 - betp[y == 1]), 'normal_supervised': np.sort(betp[y == 0]),
                'normal_typicality': np.sort(share[y == 0]), 'epsilon': float(epsilon), 'scores': ORTHOGONAL}
    if betp_normal is None:
        return {'attack': np.sort(1 - betp[y == 1]), 'normal': np.sort(betp[y == 0]), 'epsilon': float(epsilon)}
    betp_normal = _aligned_vector(betp_normal, 'betp_normal', len(betp))
    return {'attack': np.sort(1 - betp[y == 1]), 'normal': np.sort(1 - betp_normal[y == 0]),
            'epsilon': float(epsilon), 'scores': CLASS_SPECIFIC}


def _p_values(sorted_cal, alpha):
    sorted_cal = _finite_vector(sorted_cal, 'sorted_cal')
    alpha = _finite_vector(alpha, 'alpha')
    if np.any(sorted_cal[1:] < sorted_cal[:-1]):
        raise ValueError('共形校准分数必须按非递减顺序排列')
    greater_equal = len(sorted_cal) - np.searchsorted(sorted_cal, alpha, side='left')
    return (greater_equal + 1) / (len(sorted_cal) + 1)


def predict_conformal(params, betp, epsilon=None, betp_normal=None, novelty_share=None):
    betp = _finite_vector(betp, 'betp')
    eps = _epsilon(params['epsilon'] if epsilon is None else epsilon)
    extra = {}
    if params.get('scores') == ORTHOGONAL:
        if novelty_share is None:
            raise ValueError('该共形参数需要新异占比 Bel(U) / (Bel(U) + Bel(N))')
        share = _aligned_vector(novelty_share, 'novelty_share', len(betp))
        extra = {'cp_p_normal_supervised': _p_values(params['normal_supervised'], betp),
                 'cp_p_normal_typicality': _p_values(params['normal_typicality'], share)}
        p_normal = np.minimum(1.0, 2 * np.minimum(extra['cp_p_normal_supervised'], extra['cp_p_normal_typicality']))
        alpha_normal = None
    elif params.get('scores') == CLASS_SPECIFIC:
        if betp_normal is None:
            raise ValueError('该共形参数需要 BetP(正常)')
        alpha_normal = 1 - _aligned_vector(betp_normal, 'betp_normal', len(betp))
    elif 'scores' in params:
        raise ValueError('unknown conformal score layout: ' + str(params['scores']))
    else:
        alpha_normal = betp
    p_attack = _p_values(params['attack'], 1 - betp)
    if alpha_normal is not None:
        p_normal = _p_values(params['normal'], alpha_normal)
    has_attack, has_normal = p_attack > eps, p_normal > eps
    label = np.where(has_attack & ~has_normal, ATTACK, np.where(has_normal & ~has_attack, NORMAL, UNSURE))
    return {'cp_p_attack': p_attack, 'cp_p_normal': p_normal, 'cp_set': label,
            'cp_set_size': has_attack.astype(int) + has_normal.astype(int), **extra}


def _finite_vector(values, name):
    """Reject column-vector broadcasting and nonnumeric/nonfinite input explicitly."""
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(name + ' 不得包含被掩码隐藏的缺失值')
    try:
        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind not in 'biuf':
            raise ValueError
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' 必须是一维数值数组') from exc
    if not np.isfinite(array).all():
        raise ValueError(name + ' 必须全部为有限值')
    return array


def _aligned_vector(values, name, rows):
    array = _finite_vector(values, name)
    if len(array) != rows:
        raise ValueError(name + ' 必须与 betp 等长')
    return array


def _epsilon(value):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('epsilon 必须是 (0,1) 内的有限数值')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('epsilon 必须是 (0,1) 内的有限数值') from exc
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError('epsilon 必须是 (0,1) 内的有限数值')
    return value


def _weights(values, name):
    array = _finite_vector(values, name)
    if (array < 0).any():
        raise ValueError(name + ' 不得为负')
    return array


def _integer_weights(weights):
    """Represent finite float64 weights exactly over one power-of-two denominator."""
    ratios = [float(value).as_integer_ratio() for value in weights]
    powers = [denominator.bit_length() - 1 for _, denominator in ratios]
    common_power = max(powers, default=0)
    integers = [numerator << (common_power - power)
                for (numerator, _), power in zip(ratios, powers)]
    return integers, common_power


def _exact_weighted_ratios(tails, total, common_power, test_weights):
    """Round only the final exact integer ratio; no scale-dependent rounding or epsilon slack."""
    result = np.empty(len(test_weights), dtype=np.float64)
    for i, (tail, weight) in enumerate(zip(tails, test_weights)):
        numerator, denominator = float(weight).as_integer_ratio()
        shift = denominator.bit_length() - 1 - common_power
        if shift > 0:
            top, bottom = (tail << shift) + numerator, (total << shift) + numerator
        else:
            numerator <<= -shift
            top, bottom = tail + numerator, total + numerator
        if bottom == 0:
            raise ValueError('校准总权重与查询自重之和必须大于 0')
        result[i] = top / bottom
    return result


def weighted_p_floor(cal_weights, test_weights):
    """Exact-weight lower bound w(x)/(sum(w_i)+w(x)), shared by B2 screening and prediction."""
    w, wt = _weights(cal_weights, 'cal_weights'), _weights(test_weights, 'test_weights')
    integers, common_power = _integer_weights(w)
    return _exact_weighted_ratios([0] * len(wt), sum(integers), common_power, wt)


def weighted_p_values(cal_scores, cal_weights, scores, test_weights):
    """Weighted split-conformal p-values (Tibshirani et al. 2019).

    p(x) = (sum of w_i over calibration scores s_i >= s(x) + w(x)) / (sum of w_i + w(x)).
    Finite float64 weights are summed as exact binary integers. This avoids both
    overflow/underflow of intermediate sums and false threshold crossings from
    normalising by different scales. Only the final ratio is rounded to float64.
    Empty/zero-weight calibration gives p=1 for a positive query weight; a zero
    denominator is undefined and rejected. No implicit vector broadcasting.
    """
    cal, s = _finite_vector(cal_scores, 'cal_scores'), _finite_vector(scores, 'scores')
    w, wt = _weights(cal_weights, 'cal_weights'), _weights(test_weights, 'test_weights')
    if len(cal) != len(w) or len(s) != len(wt):
        raise ValueError('校准分数/权重及查询分数/权重必须分别等长')
    order = np.argsort(cal, kind='stable')
    cal = cal[order]
    integers, common_power = _integer_weights(w[order])
    suffix = [0] * (len(w) + 1)
    for i in range(len(w) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + integers[i]
    positions = np.searchsorted(cal, s, side='left')
    return _exact_weighted_ratios((suffix[i] for i in positions), suffix[0], common_power, wt)


def weighted_sets(cal, eps, betp, novelty_share, test_weights):
    """Orthogonal-test conformal sets with weights; cal holds the unsorted calibration scores and their weights:
    {'attack': (scores, w), 'normal_supervised': (scores, w), 'normal_typicality': (scores, w)}."""
    betp, share = _finite_vector(betp, 'betp'), _finite_vector(novelty_share, 'novelty_share')
    if len(betp) != len(share):
        raise ValueError('betp 与 novelty_share 必须等长')
    if not np.isscalar(eps) or not isinstance(eps, (int, float, np.integer, np.floating)) or not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('epsilon 必须是 (0,1) 内的有限数值')
    p_attack = weighted_p_values(*cal['attack'], 1 - betp, test_weights)
    p_sup = weighted_p_values(*cal['normal_supervised'], betp, test_weights)
    p_typ = weighted_p_values(*cal['normal_typicality'], share, test_weights)
    p_normal = np.minimum(1.0, 2 * np.minimum(p_sup, p_typ))
    has_attack, has_normal = p_attack > eps, p_normal > eps
    label = np.where(has_attack & ~has_normal, ATTACK, np.where(has_normal & ~has_attack, NORMAL, UNSURE))
    return {'cp_p_attack': p_attack, 'cp_p_normal': p_normal, 'cp_set': label,
            'cp_set_size': has_attack.astype(int) + has_normal.astype(int),
            'cp_p_normal_supervised': p_sup, 'cp_p_normal_typicality': p_typ}
