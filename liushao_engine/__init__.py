"""Trusted triage engine (1.3.7, format 2 since 1.3.8): evidence fusion and uncertainty on top of the frozen base learners.

Chain: heterogeneous detectors -> Dempster-Shafer fusion -> Mondrian conformal sets ->
Venn-Abers interval -> tree-path attribution -> MMD drift.
The frozen RF/LR bundle and its hash-locked sources are never modified; this layer
only appends columns.  Every caller falls back to the 1.3.6 behaviour on failure.

Engine formats: 1 = the frozen 1.3.7 parameters (step isotonic calibrators,
version hash over four values); 2 = 1.3.8 (centred isotonic calibrators,
precomputed exact Venn-Abers table, version hash over every stored parameter,
optional benchmark protocol SHA).  Both formats load; format 1 output is unchanged.

Evidence frames: 'legacy' = RF, LR and ECOD on {attack, normal} (format 1, and the first
format 2 fit); 'open' (format 2 from T07) = RF and LR as attack detectors plus the T06
typicality expert on {A known attack, N normal, U unseen}, see liushao_engine.evidence.
The open frame calibrates on the protocol's three-way split of validation half B
(B1 Venn-Abers, B2 conformal, B3 kept for risk control) and uses class-specific conformal
scores, so a record that is neither normal-like nor known-attack-like gets the empty set.
An engine is on the open frame when its fusion parameters carry 'frame'.
"""
import hashlib
import json
import math
from copy import copy
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from liushao.model import ranked, review_ranked

from .attribution import contributions, tree_paths
from . import drift
from .conformal import (ATTACK, NORMAL, ORTHOGONAL, UNSURE, _finite_vector, _integer_weights, _weights, fit_conformal,
                        predict_conformal, weighted_p_floor, weighted_sets)
from .ecod import _feature_matrix, _sum_tail_logs, fit_ecod, score_ecod, tail_logs
from .evidence import (ATYPICAL_FOCAL, THETA, bpa, calibrate, fit_atypical_map, fit_calibrator, fuse, fuse_open,
                       novelty_share, reliability, source_mass, validated_scores)
from .evidence import A as FOCAL_A, N as FOCAL_N, U as FOCAL_U
from .mmd import mmd_test
from .typicality import _score_typicality_from_tails, fit_typicality, score_typicality
from .venn_abers import fit_exact_table, fit_va, predict_va

ENGINE_FORMAT = 2
SUPPORTED_FORMATS = (1, 2)
SOURCES = ('rf', 'lr', 'ecod')
COLUMNS = ['ecod_score', 'ds_belief', 'ds_plausibility', 'ds_conflict', 'ds_betp', 'cp_set', 'cp_set_size',
           'cp_p_attack', 'cp_p_normal', 'va_low', 'va_high', 'engine_review', 'engine_version']
# Open-world fields (snapshot schema 2).  Later 1.3.8 stages add these columns; a
# record without them stores None, it is never back-filled or guessed.
OPEN_WORLD_FIELDS = {'state': 'ow_state', 'bel_unknown': 'ds_bel_unknown', 'drift_delta': 'drift_delta',
                     'alpha': 'risk_alpha', 'guarantee_status': 'guarantee_status', 'cluster_id': 'novelty_cluster'}
# Open evidence frame (T07): experts, extra columns and the settings fixed in configs/t07_rules.json.
FRAMES = ('legacy', 'open')
OPEN_SOURCES = ('rf', 'lr', 'typ')
OPEN_KINDS = {'rf': 'supervised', 'lr': 'supervised', 'typ': 'typicality'}
OPEN_COLUMNS = ['typ_score', 'ds_bel_normal', 'ds_bel_unknown', 'ds_betp_normal', 'ds_betp_unknown']
TYP_ALPHA = 0.05
TYP_RELIABILITY = 0.95
B_SPLIT_SEED = 138
# Normal-class conformal on the open frame: 'orthogonal' = one test per expert group (configs/t07b_rules.json),
# 'joint' = the single score 1 - BetP(N) of the first T07 fit.
NORMAL_TESTS = ('orthogonal', 'joint')
VA_DECIMALS = 12
FOUR_STATE_CALIBRATION_ERROR = '四态规则不支持 joint 校准，需要 orthogonal 校准的两个正常类检验。'


def _require_four_state_calibration(states, orthogonal):
    if states == STATE_RULES and not orthogonal:
        raise ValueError(FOUR_STATE_CALIBRATION_ERROR)


def validate_engine_compatibility(engine):
    """Validate combinations of parameters stored in a fitted engine."""
    _require_four_state_calibration(engine.get('states'), engine.get('conformal', {}).get('scores') == ORTHOGONAL)


def _binary_labels(values, name, rows=None):
    values = _finite_vector(values, name)
    if (rows is not None and len(values) != rows) or not np.isin(values, [0, 1]).all():
        raise ValueError(name + ' must be aligned binary labels (0 or 1)')
    return values.astype(int, copy=False)


def half_b_thirds(y_b, seed=B_SPLIT_SEED):
    """Protocol calibration_data.half_b_three_way_split: 0 = B1 (Venn-Abers), 1 = B2 (conformal), 2 = B3 (risk control)."""
    y_b = _binary_labels(y_b, 'half B labels')
    rng = np.random.default_rng(seed)
    part = np.empty(len(y_b), dtype=int)
    for label in (0, 1):
        for k, rows in enumerate(np.array_split(rng.permutation(np.flatnonzero(y_b == label)), 3)):
            part[rows] = k
    return part


def fit_engine(bundle, x_train, y_train, x_cal, y_cal, *, epsilon=0.05, conflict_quantile=0.99, seed=42,
               engine_format=ENGINE_FORMAT, protocol_sha256=None, frame=None, typ_reliability=TYP_RELIABILITY,
               typ_alpha=TYP_ALPHA, atypical_focal='AU', normal_tests='orthogonal', va_decimals=VA_DECIMALS,
               cal_blocks=None, drift_discount=None, conflict_gate=None, weighting=None, states=None,
               risk_control=None, events=None, guarantee_applicability=False):
    """Fit on training normals (ECOD / typicality) and two disjoint halves of validation.

    Half A fits the isotonic calibrators, source reliabilities and conflict cut;
    half B is held for conformal and Venn-Abers calibration, so the conformal
    calibration scores are exchangeable with new data.  engine_format=1
    reproduces the frozen 1.3.7 fit exactly.  frame: 'legacy' or 'open' (default:
    legacy for format 1, open for format 2); the typ_* and atypical_focal settings
    only apply to the open frame.  cal_blocks (open frame only): the validation time blocks;
    given, the output-space drift model is fitted on half A (T09) and drift_discount
    ({'form': 'none' | 'per_expert' | 'uniform', 'lambda': float}) sets the batch discount.
    Without cal_blocks the fit is parameter-for-parameter the T07b engine.  conflict_gate (open
    frame, T12a): 'none' | 'q99' | 'q95' sets the conflict-gated queue and its half-A quantile;
    None keeps the T07 queue rule (K >= the conflict_quantile cut) and the earlier parameters.
    weighting (open frame with drift model, T10): 'none' or a weight cap c for the output-space
    density-ratio weighted conformal on drifting batches; None keeps the earlier parameters.
    states (open frame, T11): STATE_RULES enables the four-state output and its queue / rank.
    risk_control (T12): 'class' (protocol main) or 'global' (ablation) conformal risk control on B3;
    events (O1): {'k': signature features, 'reps': representatives or None (cluster display only)}.
    """
    if guarantee_applicability and risk_control is None:
        raise ValueError('保证适用性披露需要风险控制')
    if engine_format not in SUPPORTED_FORMATS:
        raise ValueError('unsupported engine format')
    frame = frame or ('legacy' if engine_format == 1 else 'open')
    if frame not in FRAMES or (frame == 'open' and engine_format < 2):
        raise ValueError('unsupported evidence frame for this engine format')
    if frame == 'open':
        _require_four_state_calibration(states, normal_tests != 'joint')
    kind = 'step' if engine_format == 1 else 'centred'
    _feature_matrix(x_train, 'x_train', require_rows=True)
    _feature_matrix(x_cal, 'x_cal', require_rows=True)
    y_train = _binary_labels(y_train, 'training labels', len(x_train))
    y_cal = _binary_labels(y_cal, 'calibration labels', len(x_cal))
    ecod = fit_ecod(np.asarray(x_train)[y_train == 0], seed=seed)
    rng = np.random.default_rng(seed)
    half_a = np.zeros(len(y_cal), dtype=bool)
    for label in (0, 1):  # stratified 50/50 split of validation
        idx = np.flatnonzero(y_cal == label)
        half_a[rng.choice(idx, len(idx) // 2, replace=False)] = True
    if conflict_gate is not None:
        if conflict_gate not in CONFLICT_GATES or frame != 'open':
            raise ValueError('冲突门控设定无效: ' + repr(conflict_gate))
        conflict_quantile = CONFLICT_GATES[conflict_gate]
    if frame == 'open':
        engine = _fit_open(bundle, x_train, y_train, x_cal, y_cal, ecod, half_a, epsilon=epsilon,
                         conflict_quantile=conflict_quantile, seed=seed, engine_format=engine_format,
                         protocol_sha256=protocol_sha256, typ_reliability=typ_reliability, typ_alpha=typ_alpha,
                         atypical_focal=atypical_focal, normal_tests=normal_tests, va_decimals=va_decimals,
                         cal_blocks=cal_blocks, drift_discount=drift_discount, store_b3=risk_control is not None)
        if conflict_gate is not None:
            engine['conflict_gate'] = conflict_gate
        if weighting is not None:
            if 'drift' not in engine:
                raise ValueError('加权共形需要漂移模型（cal_blocks）')
            engine['weighting'] = _weighting_settings(weighting)
        if states is not None:
            if states != STATE_RULES:
                raise ValueError('四态规则版本无效: ' + repr(states))
            engine['states'] = states
        if risk_control is not None:
            if risk_control not in RISK_MODES or engine.get('states') != STATE_RULES or 'drift' not in engine:
                raise ValueError('风险控制需要四态与漂移模型，模式为 class 或 global: ' + repr(risk_control))
            engine['risk_control'] = _fit_risk_control(engine, risk_control)
            if guarantee_applicability:
                engine['risk_control']['applicability'] = 'contract-7'
        else:
            engine.pop('risk_scores', None)
        if events is not None:
            if engine.get('states') != STATE_RULES or set(events) != {'k', 'reps'} or int(events['k']) < 1:
                raise ValueError('事件归并设定无效: ' + repr(events))
            engine['events'] = {'k': int(events['k']), 'reps': None if events['reps'] is None else int(events['reps']),
                                'signature': 'state + top-k typicality tail features', 'seed': EVENT_SEED}
        return engine
    if cal_blocks is not None:
        raise ValueError('漂移模型只用于开放证据框架')
    scores = base_scores(bundle, ecod, x_cal)
    calibrators = {s: fit_calibrator(scores[s][half_a], y_cal[half_a], kind=kind) for s in SOURCES}
    fusion = {'sources': list(SOURCES), 'calibrators': calibrators,
              'reliability': {s: reliability(calibrate(calibrators[s], scores[s][half_a]), y_cal[half_a])
                              for s in SOURCES}}
    fused_a = fuse({s: v[half_a] for s, v in scores.items()}, fusion)
    fused_b = fuse({s: v[~half_a] for s, v in scores.items()}, fusion)
    conflict_cut = float(np.quantile(fused_a['ds_conflict'], conflict_quantile))
    conformal = fit_conformal(fused_b['ds_betp'], y_cal[~half_a], epsilon)
    engine = {'format': engine_format, 'model_version': bundle['model_version'], 'ecod': ecod, 'fusion': fusion,
              'conformal': conformal,
              'venn_abers': fit_va(fused_b['ds_betp'], y_cal[~half_a]),
              'conflict_cut': conflict_cut, 'conflict_quantile': conflict_quantile,
              'drift_reference': drift_space(bundle, x_train.sample(min(2000, len(x_train)), random_state=seed)),
              'split': {'ecod_rows': int(min(len(ecod['sorted']), (y_train == 0).sum())),
                        'calibration_a': int(half_a.sum()), 'calibration_b': int((~half_a).sum()),
                        'calibration_b_attacks': int(y_cal[~half_a].sum())}}
    if engine_format >= 2:
        engine['calibrator_kind'] = kind
        engine['venn_abers_exact'] = fit_exact_table(conformal['normal'], 1 - conformal['attack'])
        engine['benchmark_protocol_sha256'] = protocol_sha256
    return engine


def _fit_open(bundle, x_train, y_train, x_cal, y_cal, ecod, half_a, *, epsilon, conflict_quantile, seed,
              engine_format, protocol_sha256, typ_reliability, typ_alpha, atypical_focal, normal_tests, va_decimals,
              cal_blocks=None, drift_discount=None, store_b3=False):
    """Open-frame fit (T07; drift model T09).  Nothing here reads a development or test set."""
    if not 0 <= typ_reliability <= 1:
        raise ValueError('典型性专家的可靠度必须在 [0, 1] 内')
    if normal_tests not in NORMAL_TESTS:
        raise ValueError('unknown normal-class test layout: ' + str(normal_tests))
    if drift_discount is not None and cal_blocks is None:
        raise ValueError('漂移折扣需要验证集的时间块（cal_blocks）')
    if cal_blocks is not None:
        cal_blocks = drift._block_vector(cal_blocks)
        if len(cal_blocks) != len(y_cal):
            raise ValueError('cal_blocks 与验证集行数不一致')
    typicality = fit_typicality(np.asarray(x_train, dtype=np.float64)[y_train == 0], seed=seed)
    tail = typicality.pop('tail')  # the same 20,000 reference rows as ECOD: stored once, as engine['ecod']
    if not (np.array_equal(tail['sorted'], ecod['sorted']) and np.array_equal(tail['median'], ecod['median'])):
        raise RuntimeError('典型性参考集与 ECOD 参考集不一致')
    partial = {'ecod': ecod, 'typicality': typicality}
    scores = expert_scores(bundle, partial, x_cal)
    y_a, normal_a = y_cal[half_a], half_a & (y_cal == 0)
    calibrators = {s: fit_calibrator(scores[s][half_a], y_a, kind='centred') for s in ('rf', 'lr')}
    rel = {s: reliability(calibrate(calibrators[s], scores[s][half_a]), y_a) for s in ('rf', 'lr')}
    rel['typ'] = float(typ_reliability)
    fusion = {'frame': 'ANU', 'theta': int(THETA), 'sources': list(OPEN_SOURCES), 'kinds': dict(OPEN_KINDS),
              'calibrators': calibrators, 'reliability': rel,
              'atypical_map': fit_atypical_map(scores['typ'][normal_a], typ_alpha), 'atypical_focal': atypical_focal}
    part = np.full(len(y_cal), -1)
    part[~half_a] = half_b_thirds(y_cal[~half_a])
    settings = {'epsilon': epsilon, 'conflict_quantile': conflict_quantile, 'normal_tests': normal_tests,
                'va_decimals': va_decimals}
    calibrated = _calibrate_open(fusion, scores, y_cal, part, settings)
    b1 = part == 0
    count = lambda mask: {'normal': int((mask & (y_cal == 0)).sum()), 'attack': int((mask & (y_cal == 1)).sum())}
    engine = {'format': engine_format, 'model_version': bundle['model_version'], 'ecod': ecod, 'typicality': typicality,
              'fusion': fusion, **calibrated,
              'conflict_quantile': conflict_quantile,
              'drift_reference': drift_space(bundle, x_train.sample(min(2000, len(x_train)), random_state=seed)),
              'split': {'ecod_rows': int(min(len(ecod['sorted']), (y_train == 0).sum())),
                        'typicality_calibration_rows': int(len(typicality['calibration'])),
                        'calibration_a': int(half_a.sum()), 'atypical_null_rows': int(normal_a.sum()),
                        'calibration_b': int((~half_a).sum()), 'calibration_b_attacks': int(y_cal[~half_a].sum()),
                        'b1_venn_abers': count(b1), 'b2_conformal': count(part == 1), 'b3_risk_control_unused': count(part == 2)},
              'calibrator_kind': 'centred',
              'benchmark_protocol_sha256': protocol_sha256}
    if cal_blocks is not None:
        z = np.column_stack([scores[s] for s in drift.EXPERTS])
        engine['drift'] = drift.fit_drift(z[half_a], y_a, cal_blocks[half_a], rf_threshold=float(bundle['config']['rf_threshold']),
                                          candidates=(drift.SELECTED,))
        keep = part != 2  # B3 belongs to risk control (T12) and is not stored here
        engine['calibration_scores'] = {**{s: scores[s][keep] for s in OPEN_SOURCES}, 'y': y_cal[keep], 'part': part[keep],
                                        'settings': settings}
        engine['drift_discount'] = _discount_settings(drift_discount)
    if store_b3:  # T12: B3 is read only by risk control
        b3 = part == 2
        engine['risk_scores'] = {**{s: scores[s][b3] for s in OPEN_SOURCES}, 'y': y_cal[b3]}
    return engine


def _calibrate_open(fusion, scores, y_cal, part, settings):
    """Every calibrated quantity that depends on the fused output, from raw expert scores (part: -1 A, 0 B1, 1 B2)."""
    fused = fuse_open(scores, fusion)
    half_a, b1, b2 = part == -1, part == 0, part == 1
    betp = fused['ds_betp']
    if settings['normal_tests'] == 'orthogonal':
        share = novelty_share(fused['ds_bel_unknown'], fused['ds_bel_normal'])
        conformal = fit_conformal(betp[b2], y_cal[b2], settings['epsilon'], novelty_share=share[b2])
    else:
        conformal = fit_conformal(betp[b2], y_cal[b2], settings['epsilon'], betp_normal=fused['ds_betp_normal'][b2])
    # Keep the registered B1 rounding. Queries within sklearn's pooling resolution
    # still need the application's exact near-point refit using these stored B1 scores.
    decimals = settings['va_decimals']
    va_scores = betp if decimals is None else np.round(betp, decimals)
    va_cal = {'normal': np.sort(va_scores[b1 & (y_cal == 0)]), 'attack': np.sort(1 - va_scores[b1 & (y_cal == 1)])}
    return {'conformal': conformal, 'venn_abers': fit_va(betp[b1], y_cal[b1]), 'venn_abers_calibration': va_cal,
            'conflict_cut': float(np.quantile(fused['ds_conflict'][half_a], settings['conflict_quantile'])),
            'venn_abers_exact': fit_exact_table(va_cal['normal'], 1 - va_cal['attack'])}


DISCOUNT_FORMS = ('none', 'per_expert', 'uniform')
# T12a (configs/t12a_rules.json): conflict-gated queue; the value is the half-A quantile of K used as the cut
# ('none' keeps the default cut in the parameters but never gates).
CONFLICT_GATES = {'none': 0.99, 'q99': 0.99, 'q95': 0.95}


def conflict_gate(engine, conflict):
    """Records queued because RF and LR disagree.  Engines fitted before T12a keep K >= cut; T12a engines
    gate only when the cut is positive (a cut of 0 means K vanished on half A, e.g. a fully discounted source)."""
    conflict = np.asarray(conflict, dtype=np.float64)
    mode = engine.get('conflict_gate')
    if mode is None:
        return conflict >= engine['conflict_cut']
    if mode == 'none':
        return np.zeros(len(conflict), dtype=bool)
    return (conflict >= engine['conflict_cut']) & (engine['conflict_cut'] > 0)
DRIFT_COLUMNS = ['drift_flag', 'drift_delta', 'drift_delta_rf', 'drift_delta_lr', 'drift_delta_typ']


def _discount_settings(value):
    value = dict(value or {'form': 'none', 'lambda': 0.0})
    if value.get('form') not in DISCOUNT_FORMS or not float(value.get('lambda', -1)) >= 0 or set(value) - {'form', 'lambda'}:
        raise ValueError('漂移折扣设定无效: ' + repr(value))
    return {'form': value['form'], 'lambda': float(value['lambda']), 'gate': 'batch_drift_flag'}


def with_discount(engine, form, lam):
    """Copy of a drift-aware engine with another discount setting (for ablations and the T09 grid)."""
    if 'drift' not in engine:
        raise ValueError('该引擎没有漂移模型')
    return {**engine, 'drift_discount': _discount_settings({'form': form, 'lambda': lam})}


def discount_factors(setting, detected):
    """Reliability multipliers clip(1 - lambda * delta_s, 0, 1); all 1 unless the whole batch drifts (configs/t09_rules.json)."""
    ones = {s: 1.0 for s in OPEN_SOURCES}
    if setting['form'] == 'none' or setting['lambda'] == 0 or not detected or not detected.get('drift'):
        return ones
    delta = {s: detected['delta_s'][s] if setting['form'] == 'per_expert' else detected['delta'] for s in OPEN_SOURCES}
    return {s: float(np.clip(1 - setting['lambda'] * delta[s], 0.0, 1.0)) for s in OPEN_SOURCES}


def batch_engine(engine, scores):
    """(engine used for this batch, drift result or None).

    One drift test per batch on the expert outputs; when the batch drifts and a discount is set, the
    reliabilities are discounted and every calibrated quantity is recomputed under the same discount
    from the stored calibration scores, so calibration and test scores come from one function (T09).
    """
    if 'drift' not in engine:
        return engine, None
    scores = validated_scores(scores, OPEN_SOURCES)
    z = np.column_stack([scores[s] for s in drift.EXPERTS])
    detected = drift.detect_drift(engine['drift'], z) if len(z) else {'status': 'too_small', 'drift': None}
    factors = discount_factors(engine['drift_discount'], detected)
    detected = {**detected, 'factors': factors}
    if all(f == 1.0 for f in factors.values()):
        return engine, detected
    fusion = {**engine['fusion'], 'reliability': {s: engine['fusion']['reliability'][s] * factors[s] for s in OPEN_SOURCES}}
    cal = engine['calibration_scores']
    calibrated = _calibrate_open(fusion, {s: cal[s] for s in OPEN_SOURCES}, cal['y'], cal['part'], cal['settings'])
    return {**engine, 'fusion': fusion, **calibrated, '_batch_drift': detected}, detected


T10_V2 = 't10-v2'
B2_COLUMNS = ['b2_mode', 'b2_fallback_reason']
B3_COLUMNS = ['b3_calculation_status', 'b3_statistical_status']


def _optional_columns(engine):
    """Contract columns that only engines fitted by later stages produce."""
    return ((DRIFT_COLUMNS if 'drift' in engine else []) + (['ow_conflict_gate'] if 'conflict_gate' in engine else []) +
            (['guarantee_ess'] if 'weighting' in engine else []) + (STATE_COLUMNS if 'states' in engine else []) +
            (RISK_COLUMNS if 'risk_control' in engine else []) + (EVENT_COLUMNS if 'events' in engine else []) +
            ((B2_COLUMNS + (B3_COLUMNS if 'risk_control' in engine else []))
             if engine.get('weighting', {}).get('policy') == T10_V2 else []))


# T12 (configs/t12_rules.json): class-conditional conformal risk control on B3 and the batch guarantee status.
RISK_ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_ALPHA = 0.05
NOVELTY_CAP = 0.10
RISK_MODES = ('class', 'global')
RISK_COLUMNS = ['risk_alpha', 'guarantee_status']
GUARANTEE = ('有效', '加权有效', '失效', '不适用')


def crc_threshold(betp, state, y, alpha, weights=None, test_weight=1.0, mode='class'):
    """Largest candidate t with the exact binary64-input CRC bound <= alpha.

    Loss is 1 only for known attacks in state 正常 with BetP(A) < t.  Class
    mode divides by attack weight; global mode includes every calibration row.
    Integer accumulation prevents overflow and gate-changing sum/ratio rounding.
    Tied scores are removed together before checking their strict threshold.
    Only the returned diagnostic bound is rounded to float64.
    """
    if not isinstance(mode, (str, np.str_)) or mode not in RISK_MODES:
        raise ValueError('CRC mode 必须为 class 或 global')
    scalar_types = (int, float, np.integer, np.floating)
    for name, value in (('alpha', alpha), ('test_weight', test_weight)):
        try:
            valid = (not isinstance(value, (bool, np.bool_)) and isinstance(value, scalar_types)
                     and math.isfinite(float(value)))
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            raise ValueError('CRC ' + name + ' 必须为有限数值标量')
    alpha, test_weight = float(alpha), float(test_weight)
    if not 0 <= alpha <= 1 or test_weight < 0:
        raise ValueError('CRC alpha 必须在 [0,1] 内，查询自重不得为负')
    betp, y = _finite_vector(betp, 'CRC betp'), _finite_vector(y, 'CRC labels')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('CRC labels 只能为 0 或 1')
    if np.ma.isMaskedArray(state) and np.ma.getmaskarray(state).any():
        raise ValueError('CRC state 不得包含被掩码隐藏的缺失值')
    state = np.asarray(state)
    if state.ndim != 1 or any(not isinstance(v, (str, np.str_)) or v not in STATES for v in state):
        raise ValueError('CRC state 必须为一维四态数组')
    w = np.ones(len(y)) if weights is None else _weights(weights, 'CRC weights')
    if len(betp) != len(y) or len(state) != len(y) or len(w) != len(y):
        raise ValueError('CRC 分数、状态、标签和权重必须等长')
    integers, _ = _integer_weights(np.r_[w, test_weight])
    self_weight, integers = integers[-1], integers[:-1]
    pool = (y == 1) if mode == 'class' else np.ones(len(y), dtype=bool)
    total = sum(value for value, keep in zip(integers, pool) if keep) + self_weight
    if not total:
        raise ValueError('CRC 校准池总权重与查询自重之和必须大于 0')
    exposed = np.flatnonzero((y == 1) & (state == '正常'))
    loss = sum(integers[i] for i in exposed)
    alpha_num, alpha_den = alpha.as_integer_ratio()

    def accepted(value):
        return (value + self_weight) * alpha_den <= total * alpha_num

    if accepted(loss):
        return float('inf'), float((loss + self_weight) / total)
    order = exposed[np.argsort(betp[exposed], kind='stable')]
    end = len(order)
    while end:
        threshold = betp[order[end - 1]]
        start = end - 1
        while start and betp[order[start - 1]] == threshold:
            start -= 1
        loss -= sum(integers[i] for i in order[start:end])
        if accepted(loss):
            return float(threshold), float((loss + self_weight) / total)
        end = start
    return None, float(self_weight / total)


def _fit_risk_control(engine, mode):
    b3 = engine.pop('risk_scores')
    out = analyze_scores({s: b3[s] for s in OPEN_SOURCES}, engine, _batch=True)
    thresholds = [[a, crc_threshold(out['ds_betp'], out['ow_state'], b3['y'], a, mode=mode)[0]] for a in RISK_ALPHAS]
    return {'mode': mode, 'b3': b3, 'thresholds': thresholds, 'default_alpha': DEFAULT_ALPHA, 'novelty_cap': NOVELTY_CAP,
            'score': 'BetP(A) among records in state 正常', 'b3_rows': {'normal': int((b3['y'] == 0).sum()),
                                                                     'attack': int((b3['y'] == 1).sum())}}


def risk_table(used, out, weights, drifting, *, drift_checked=True, b2_weighted=True):
    """Per alpha: threshold, bound, guarantee status, risk-queue size for this batch (configs/t12_rules.json)."""
    rc = used['risk_control']
    b3 = rc['b3']
    b3_scores = {s: b3[s] for s in OPEN_SOURCES}
    plain = analyze_scores(b3_scores, used, _batch=True)
    state = np.asarray(out['ow_state'])
    novelty = float(np.mean(state == '新异')) if len(state) else 0.0
    uncertain = float(np.mean(np.isin(state, ['模糊', '新异']))) if len(state) else 0.0
    known_attack_evidence = bool((state == '攻击').any())
    out_of_scope = rc.get('applicability') == 'contract-7' and (
        uncertain >= 0.5 or not known_attack_evidence or not drift_checked
    )
    v2 = used.get('weighting', {}).get('policy') == T10_V2
    weighted = weights is not None and 'b3' in weights
    weight_status = 'not_estimated'
    if v2 and weighted:
        try:
            w = _weights(weights['b3'], 'B3 weights')
            if len(w) != len(b3['y']):
                raise ValueError('B3 权重与校准标签必须等长')
            pool = np.asarray(b3['y']) == 1 if rc['mode'] == 'class' else np.ones(len(w), dtype=bool)
            weighted = bool(np.any(w[pool] > 0))
        except (ValueError, TypeError, IndexError):
            weighted = False
        weight_status = 'usable_for_diagnostics' if weighted else 'invalid'
    if weighted:
        if not v2:
            w = weights['b3']
        b3_out = (analyze_scores(b3_scores, used, _batch=True, _weights={'table': weights['table'], 'test': w})
                  if b2_weighted else plain)
        ess_attack = ess(w[b3['y'] == 1])
    rows, released = [], state == '正常'
    for a in RISK_ALPHAS:
        matched = b3_out if v2 and weighted and b2_weighted else plain
        t_plain, bound_plain = crc_threshold(matched['ds_betp'], matched['ow_state'], b3['y'], a, mode=rc['mode'])
        row = {'alpha': a, 'threshold_unweighted': t_plain, 'bound_unweighted': bound_plain, 'novelty_share': novelty,
               'drifting': drifting, 'weighted': weighted}
        if weighted:
            t_w, bound_w = crc_threshold(b3_out['ds_betp'], b3_out['ow_state'], b3['y'], a, weights=w,
                                         test_weight=weights['cap'], mode=rc['mode'])
            row.update(threshold_weighted=t_w, bound_weighted=bound_w, ess_b3_attack=ess_attack,
                       ess_required=int(np.ceil(1 / a)) - 1)
            ok = t_w is not None and ess_at_least(w[b3['y'] == 1], row['ess_required']) and novelty <= rc['novelty_cap']
            row['status'], row['threshold'] = ('加权有效' if ok else '失效'), (t_w if t_w is not None else t_plain)
        else:
            ok = not drifting and t_plain is not None and novelty <= rc['novelty_cap']
            row['status'], row['threshold'] = ('有效' if ok else '失效'), t_plain
        if v2:
            calculation_ok = ok if weighted else t_plain is not None and novelty <= rc['novelty_cap']
            statistical = ('not_applicable' if not drift_checked else
                           'weight_invalid' if weight_status == 'invalid' else
                           'premises_unverified' if drifting else 'exchangeability_assumed')
            row.update(calculation_status='passes' if calculation_ok else 'fails',
                       statistical_status=statistical, b3_weight_status=weight_status,
                       loss_path='weighted' if b2_weighted else 'unweighted')
            # Under drift the estimated ratio has no independently verified guarantee.
            if drifting or not drift_checked:
                row['status'] = '失效'
                row['threshold'] = None if b2_weighted and not weighted else t_plain
                if b2_weighted and not weighted:
                    row.update(calculation_status='unavailable', threshold_unweighted=None, bound_unweighted=None)
        if rc.get('applicability') == 'contract-7':
            row.update(display_status='不适用' if out_of_scope else row['status'],
                       uncertain_share=uncertain, known_attack_evidence=known_attack_evidence,
                       drift_checked=drift_checked)
        t = np.inf if row['threshold'] is None else row['threshold']
        row['risk_queue'] = int((released & (np.asarray(out['ds_betp']) >= t)).sum())
        rows.append(row)
    return rows


def apply_risk(out, table, alpha):
    """Risk-control queue, queue, review rank, alpha and guarantee status at one alpha."""
    if alpha not in RISK_ALPHAS:
        raise ValueError('α 只能取 ' + str(RISK_ALPHAS))
    row = next(r for r in table if r['alpha'] == alpha)
    t = np.inf if row['threshold'] is None else row['threshold']
    state, base_queue = np.asarray(out['ow_state']), np.asarray(out['ow_queue'])
    risk = (state == '正常') & (np.asarray(out['ds_betp']) >= t)
    queue = np.where(base_queue != '', base_queue, np.where(risk, '风险控制', ''))
    betp, betp_normal, label = np.asarray(out['ds_betp']), np.asarray(out['ds_betp_normal']), np.asarray(out['cp_set'])
    decision = np.where(label == ATTACK, True, np.where(label == NORMAL, False, betp >= .5))
    n = len(state)
    return {'ow_queue': queue, 'engine_review': queue != '', 'engine_rank': review_rank(queue != '', np.where(decision, betp_normal, 1 - betp_normal),
                                                                                  out['ds_conflict']),
            'risk_alpha': np.full(n, float(alpha)), 'guarantee_status': np.full(n, row.get('display_status', row['status']), dtype=object).astype(str)}


def _frame_scores(frame):
    return {'rf': frame.rf_score.to_numpy(dtype=np.float64), 'lr': frame.lr_score.to_numpy(dtype=np.float64),
            'typ': frame.typ_score.to_numpy(dtype=np.float64)}


def apply_alpha(frame, engine, alpha):
    """Recompute ow_queue, engine_review, engine_rank, risk_alpha and guarantee_status at another alpha (no base-model rerun)."""
    if 'risk_control' not in engine:
        raise ValueError('该引擎没有风险控制')
    values = analyze_batch(_frame_scores(frame), engine, alpha)[0]
    out = frame.copy()
    for col in ['ow_queue', 'engine_review', 'engine_rank', *RISK_COLUMNS, *(c for c in B3_COLUMNS if c in values)]:
        out[col] = values[col]
    return out


def alpha_for_budget(frame, engine, k):
    """Smallest alpha in RISK_ALPHAS whose whole review queue (模糊 + 新异 + 风险控制) fits in k records; None if none does."""
    if 'risk_control' not in engine:
        raise ValueError('该引擎没有风险控制')
    values, used = analyze_batch(_frame_scores(frame), engine)
    base = int(np.isin(values['ow_state'], ['模糊', '新异']).sum())
    for row in sorted(used['_risk'], key=lambda r: r['alpha']):
        if base + row['risk_queue'] <= k:
            return row['alpha']
    return None


# O1 (configs/t12_rules.json): events = queued records with the same state and top-k typicality tail features.
EVENT_SEED = 138
EVENT_COLUMNS = ['ow_event', 'novelty_cluster']


def event_ids(engine, features, state, rank):
    """Event number per record (-1 outside the 模糊 / 新异 queue); 0 is the event holding the most urgent record."""
    from .ecod import tail_logs
    state = np.asarray(state).astype(str)
    queued = np.isin(state, ['模糊', '新异'])
    ids = np.full(len(state), -1, dtype=np.int64)
    if not queued.any():
        return ids
    k = engine['events']['k']
    left, right = tail_logs(engine['ecod'], np.asarray(features, dtype=np.float64)[queued])
    top = np.sort(np.argsort(-np.maximum(left, right), axis=1, kind='stable')[:, :k], axis=1)
    keys = np.array([st + '|' + ','.join(map(str, row)) for st, row in zip(state[queued], top)])
    uniq, inverse = np.unique(keys, return_inverse=True)
    first = np.full(len(uniq), np.iinfo(np.int64).max)
    np.minimum.at(first, inverse, np.asarray(rank)[queued])
    order = np.empty(len(uniq), dtype=np.int64)
    order[np.argsort(first, kind='stable')] = np.arange(len(uniq))
    ids[queued] = order[inverse]
    return ids


def event_columns(ids, state):
    ids = np.asarray(ids)
    event = pd.array(np.where(ids >= 0, ids, 0), dtype='Int64')
    event[ids < 0] = pd.NA
    cluster = event.copy()
    cluster[np.asarray(state) != '新异'] = pd.NA
    return {'ow_event': event, 'novelty_cluster': cluster}


# T11 (configs/t11_rules.json): handoff 4.3 four states, queue and review rank.
STATE_RULES = 'handoff-4.3-v1'
STATES = ('正常', '攻击', '模糊', '新异')
STATE_COLUMNS = ['ow_state', 'ow_queue', 'ow_normal_reject', 'engine_rank', 'cp_p_normal_supervised', 'cp_p_normal_typicality']


def review_rank(queue, risk, conflict):
    """1..n: queue first, then the provisional decision's risk, then K (both descending), then input order."""
    queue = np.asarray(queue, dtype=bool)
    order = np.lexsort((np.arange(len(queue)), -np.asarray(conflict), -np.asarray(risk), ~queue))
    rank = np.empty(len(queue), dtype=np.int64)
    rank[order] = np.arange(1, len(queue) + 1)
    return rank


def four_states(out, gate, epsilon):
    """State, queue, which normal test rejected, and review rank, in the rule order of configs/t11_rules.json."""
    if not {'cp_p_normal_supervised', 'cp_p_normal_typicality'} <= out.keys():
        raise ValueError(FOUR_STATE_CALIBRATION_ERROR)
    size, label = np.asarray(out['cp_set_size']), np.asarray(out['cp_set'])
    bel = {'A': np.asarray(out['ds_belief']), 'N': np.asarray(out['ds_bel_normal']), 'U': np.asarray(out['ds_bel_unknown'])}
    leads = {k: (v > np.maximum(*[bel[o] for o in bel if o != k])) for k, v in bel.items()}  # strictly largest
    single_normal, single_attack = (size == 1) & (label == NORMAL), (size == 1) & (label == ATTACK)
    state = np.full(len(size), '模糊', dtype=object)
    rest = (size == 1) & ~np.asarray(gate, dtype=bool)  # rules 1-3 leave 新异 / 模糊 for the others
    state[rest & single_normal & leads['N']] = '正常'
    state[rest & single_attack & leads['A']] = '攻击'
    state[rest & leads['U']] = '新异'
    state[size == 0] = '新异'
    queue = np.where(state == '新异', '新异', np.where(state == '模糊', '模糊', ''))
    sup = np.asarray(out['cp_p_normal_supervised']) <= epsilon / 2
    typ = np.asarray(out['cp_p_normal_typicality']) <= epsilon / 2
    reject = np.where(sup & typ, 'both', np.where(sup, 'supervised', np.where(typ, 'typicality', '')))
    betp, betp_normal = np.asarray(out['ds_betp']), np.asarray(out['ds_betp_normal'])
    decision = np.where(label == ATTACK, True, np.where(label == NORMAL, False, betp >= .5))
    risk = np.where(decision, betp_normal, 1 - betp_normal)
    return {'ow_state': state.astype(str), 'ow_queue': queue, 'ow_normal_reject': reject,
            'engine_rank': review_rank(queue != '', risk, out['ds_conflict'])}


# T10 (configs/t10_rules.json): density-ratio weighted conformal on drifting batches.
WEIGHT_FIT_ROWS = 10000
WEIGHT_SEED = 138


def _weighting_settings(value):
    if value == 'none':
        return {'cap': None}
    cap = float(value)
    if not cap > 1:
        raise ValueError('权重截断上限必须大于 1: ' + repr(value))
    return {'cap': cap, 'gate': 'batch_drift_flag', 'space': 'z,z^2,pairwise (B2-standardised)',
            'classifier': 'LogisticRegression(C=1, balanced)', 'fit_rows': WEIGHT_FIT_ROWS, 'seed': WEIGHT_SEED}


def _quadratic(z, mu, sd):
    zs = (z - mu) / sd
    pairs = [zs[:, i] * zs[:, j] for i in range(zs.shape[1]) for j in range(i + 1, zs.shape[1])]
    return np.column_stack([zs, zs ** 2] + pairs)


def density_ratio_weights(z_cal, z_batch, cap, seed=WEIGHT_SEED, fit_rows=WEIGHT_FIT_ROWS, *, retain_raw=False):
    """w(z) = P(batch | z) / P(calibration | z) from a class-balanced logistic regression, capped at `cap`."""
    from sklearn.linear_model import LogisticRegression
    z_cal, z_batch = np.asarray(z_cal, dtype=np.float64), np.asarray(z_batch, dtype=np.float64)
    mu, sd = z_cal.mean(axis=0), z_cal.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    fit_batch = z_batch if len(z_batch) <= fit_rows else z_batch[np.random.default_rng(seed).choice(len(z_batch), fit_rows, replace=False)]
    x = _quadratic(np.vstack([fit_batch, z_cal]), mu, sd)
    y = np.r_[np.ones(len(fit_batch)), np.zeros(len(z_cal))]
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000).fit(x, y)
    odds = lambda z: np.exp(clf.decision_function(_quadratic(np.asarray(z, dtype=np.float64), mu, sd)))
    raw_cal, raw_batch = odds(z_cal), odds(z_batch)
    info = {
        'capped_share_calibration': float((raw_cal > cap).mean()), 'capped_share_batch': float((raw_batch > cap).mean()),
        'calibration_weight_quantiles': np.quantile(raw_cal, [.5, .9, .99, 1.0]).tolist()}
    if retain_raw:
        return raw_cal, raw_batch, odds, info
    return np.minimum(raw_cal, cap), np.minimum(raw_batch, cap), lambda z: np.minimum(odds(z), cap), info


def _ess_ratio(w):
    """Exact Kish numerator/denominator for validated binary64 weights."""
    integers, _ = _integer_weights(_weights(w, 'weights'))
    total = sum(integers)
    return total * total, sum(value * value for value in integers)


def ess(w):
    numerator, denominator = _ess_ratio(w)
    return float(numerator / denominator) if denominator else 0.0


def ess_at_least(w, minimum):
    """Compare before rounding: near-equal weights can round ESS up to the count."""
    numerator, denominator = _ess_ratio(w)
    target, scale = float(minimum).as_integer_ratio()
    return bool(denominator and numerator * scale >= target * denominator)


def screen_b2_weights(y, raw_cal, raw_batch, cap, epsilon):
    inputs = (y, raw_cal, raw_batch)
    if any(np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any() for value in inputs):
        raise ValueError('B2 标签和权重不得包含被掩码隐藏的缺失值')
    y, raw_cal, raw_batch = (np.asarray(value) for value in inputs)
    if any(value.dtype.kind not in 'biuf' for value in (y, raw_cal, raw_batch)):
        raise ValueError('B2 标签和权重必须为实数，不接受复数或非数值输入')
    if y.ndim != 1 or raw_cal.ndim != 1 or raw_batch.ndim != 1 or len(y) != len(raw_cal):
        raise ValueError('B2 标签、校准权重和查询权重必须为一维数组；校准标签与权重须等长')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('B2 校准标签只能为 0 或 1')
    scalar_types = (int, float, np.integer, np.floating)
    if not isinstance(cap, scalar_types) or not np.isfinite(cap) or cap <= 0:
        raise ValueError('B2 cap 必须是正的有限数值')
    if not isinstance(epsilon, scalar_types) or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError('B2 epsilon 必须是 (0,1) 内的有限数值')
    raw_cal, raw_batch = raw_cal.astype(np.float64, copy=False), raw_batch.astype(np.float64, copy=False)
    failures, report = [], {}
    valid_batch = np.isfinite(raw_batch).all() and (raw_batch >= 0).all() and len(raw_batch) > 0
    for name, label, tau in [('normal', 0, epsilon / 2), ('attack', 1, epsilon)]:
        u = raw_cal[y == label]
        count = len(u)
        minimum = int(np.ceil(1 / tau)) - 1
        row = dict(count=count, tau=tau, min_count=minimum, min_ess=minimum)
        report[name] = row
        if count < minimum:
            failures.append(name + ':count')
        if not valid_batch or not np.isfinite(u).all() or (u < 0).any():
            failures.append(name + ':weights')
            continue
        w, test = np.minimum(u, cap), np.minimum(raw_batch, cap)
        M = float(test.max())
        if not np.any(w > 0):
            failures.append(name + ':weight_sum')
            continue
        effective = ess(w)
        # Screening and prediction must share the same exact-weight arithmetic:
        # distinct normalisation scales can disagree at the registered tau.
        floor = float(weighted_p_floor(w, np.array([M]))[0])
        # D+ is diagnostic only: avoid overflowing sum(u) for valid raw
        # weights, and do not let that diagnostic force an otherwise valid
        # B2 batch to fall back. Each test point uses a safe common scale.
        scale_cal = float(u.max())
        U_scaled = math.fsum(u / scale_cal)
        removed_cal = math.fsum((u - w) / scale_cal)
        scale = np.maximum(scale_cal, raw_batch)
        cal_ratio = scale_cal / scale
        removed = removed_cal * cal_ratio + (raw_batch - test) / scale
        mass = removed / (U_scaled * cal_ratio + raw_batch / scale)
        row.update(ess=effective, p_floor=floor, delta_res=tau - floor, D_plus=float(mass.max()))
        if not ess_at_least(w, minimum):
            failures.append(name + ':ess')
        if floor > tau:
            failures.append(name + ':resolution')
    return {'classes': report, 'failures': failures}


def batch_weights(engine, scores, detected):
    """Weighted-conformal inputs for a drifting batch, or None (no weighting, or the batch does not drift)."""
    setting = engine.get('weighting')
    if not setting or setting['cap'] is None or not detected or not detected.get('drift'):
        return None
    cal = engine['calibration_scores']
    b2 = cal['part'] == 1
    z_cal = np.column_stack([cal[s][b2] for s in drift.EXPERTS])
    scores = validated_scores(scores, OPEN_SOURCES)
    z_batch = np.column_stack([scores[s] for s in drift.EXPERTS])
    cap = setting['cap']
    v2 = setting.get('policy') == T10_V2
    if v2:
        try:
            raw_cal, raw_batch, raw_of, info = density_ratio_weights(z_cal, z_batch, cap, retain_raw=True)
        except (ValueError, FloatingPointError):
            return {'info': {'classes': {}, 'failures': ['weight_fit']}}
        try:
            report = screen_b2_weights(cal['y'][b2], raw_cal, raw_batch, cap, engine['conformal']['epsilon'])
            if len(raw_batch) != len(z_batch):
                raise ValueError('B2 query weights must align with this batch')
        except (ValueError, TypeError, FloatingPointError, OverflowError):
            # An unusable estimator result is a B2 screening failure too.
            # Keep the whole batch on the existing nonweighted Mondrian path.
            return {'info': {'classes': {}, 'failures': ['weight_input']}}
        info.update(report)
        w_cal, w_batch = np.minimum(raw_cal, cap), np.minimum(raw_batch, cap)
    else:
        w_cal, w_batch, weight_of, info = density_ratio_weights(z_cal, z_batch, cap)
    fused = fuse_open({s: cal[s][b2] for s in OPEN_SOURCES}, engine['fusion'])
    y = cal['y'][b2]
    share = novelty_share(fused['ds_bel_unknown'], fused['ds_bel_normal'])
    table = {'attack': (1 - fused['ds_betp'][y == 1], w_cal[y == 1]),
             'normal_supervised': (fused['ds_betp'][y == 0], w_cal[y == 0]),
             'normal_typicality': (share[y == 0], w_cal[y == 0])}
    if v2:
        ess_normal = info['classes']['normal'].get('ess', np.nan)
        ess_attack = info['classes']['attack'].get('ess', np.nan)
    else:
        ess_normal, ess_attack = ess(w_cal[y == 0]), ess(w_cal[y == 1])
    info.update(ess_normal=ess_normal, ess_attack=ess_attack,
                calibration_normal=int((y == 0).sum()), calibration_attack=int((y == 1).sum()))
    info['ess'] = float(np.minimum(ess_normal, ess_attack))
    out = {'table': table, 'test': w_batch, 'info': info, 'cap': setting['cap']}
    if 'risk_control' in engine:  # B2 and B3 are random thirds of half B: the same density ratio applies (T12)
        z3 = np.column_stack([engine['risk_control']['b3'][s] for s in drift.EXPERTS])
        if v2:
            try:
                raw_b3 = _weights(raw_of(z3), 'B3 raw weights')
                if len(raw_b3) != len(z3):
                    raise ValueError('B3 权重估计与校准行必须等长')
                out['b3'] = np.minimum(raw_b3, cap)
            except (ValueError, TypeError, FloatingPointError, OverflowError):
                # Preserve the completed B2 decision; B3 reports its own invalid input.
                out['b3'] = np.full(len(z3), np.nan)
                info['b3_weight_error'] = 'invalid_or_misaligned_estimate'
        else:
            out['b3'] = weight_of(z3)
    return out


def drift_columns(detected, rows):
    """Batch-level contract columns, repeated on every row; batches under the smallest size class are not tested."""
    ok = detected is not None and detected.get('status') == 'ok'
    value = lambda key, s=None: (detected['delta_s'][s] if s else detected['delta']) if ok else np.nan
    return {'drift_flag': np.full(rows, bool(ok and detected['drift'])), 'drift_delta': np.full(rows, value('delta')),
            **{'drift_delta_' + s: np.full(rows, value('delta', s)) for s in drift.EXPERTS}}


def is_open(engine):
    """True for an engine fitted on the open evidence frame {A, N, U}."""
    return 'frame' in engine.get('fusion', {})


def rf_scores(bundle, x):
    """Score in fixed tree order without mutating the shared frozen estimator.

    Parallel tree accumulation changes the last bits with completion order.
    A shallow execution copy shares the fitted trees but uses one worker; for
    pipelines only the step list and final estimator need separate wrappers.
    Concurrent calls therefore cannot change each other's n_jobs or restore an
    already-overridden setting into the cached base model.
    """
    model = copy(bundle['rf'])
    if hasattr(model, 'steps'):
        name, forest = model.steps[-1]
        forest = copy(forest)
        model.steps = [*model.steps[:-1], (name, forest)]
    else:
        forest = model
    forest.n_jobs = 1
    return model.predict_proba(x)[:, 1]


def base_scores(bundle, ecod, x):
    return {'rf': rf_scores(bundle, x), 'lr': bundle['lr'].predict_proba(x)[:, 1], 'ecod': score_ecod(ecod, x)}


def typicality_scores(engine, x):
    """Atypicality -log p of the T06 signal; its tail model is the engine's ECOD reference."""
    return score_typicality(dict(engine['typicality'], tail=engine['ecod']), x)


def expert_scores(bundle, engine, x, rf=None, lr=None):
    """Raw scores of the open-frame experts; rf / lr may be passed when the frozen inference already ran."""
    return validated_scores({'rf': rf_scores(bundle, x) if rf is None else rf,
                             'lr': bundle['lr'].predict_proba(x)[:, 1] if lr is None else lr,
                             'typ': typicality_scores(engine, x)}, OPEN_SOURCES)


def analyze_batch(scores, engine, alpha=None):
    """(open-frame result columns, engine used for the batch): the batch is one drift test (T09)."""
    used, detected = batch_engine(engine, scores)
    weights = batch_weights(used, scores, detected)
    v2 = engine.get('weighting', {}).get('policy') == T10_V2
    failures = weights['info'].get('failures', []) if weights else []
    active = None if failures else weights
    out = analyze_scores(scores, used, _batch=True, _weights=active)
    if v2:
        out['b2_mode'] = np.full(len(out['ds_betp']), 'fallback' if failures else 'weighted' if active else 'unweighted')
        out['b2_fallback_reason'] = np.full(len(out['ds_betp']), ', '.join(failures))
    if detected is not None:
        out.update(drift_columns(detected, len(out['ds_betp'])))
    if 'weighting' in engine:
        out['guarantee_ess'] = np.full(len(out['ds_betp']), weights['info'].get('ess', np.nan) if weights else np.nan)
    if weights:
        used = {**used, '_weights_info': weights['info']}
    if 'risk_control' in engine:
        table = risk_table(used, out, weights, bool(detected and detected.get('drift')),
                           drift_checked=bool(detected and detected.get('status') == 'ok'), b2_weighted=active is not None)
        selected_alpha = engine['risk_control']['default_alpha'] if alpha is None else alpha
        out.update(apply_risk(out, table, selected_alpha))
        if v2:
            row = next(row for row in table if row['alpha'] == selected_alpha)
            out.update({column: np.full(len(out['ds_betp']), row[column.removeprefix('b3_')]) for column in B3_COLUMNS})
        used = {**used, '_risk': table}
    return out, used


def analyze_scores(scores, engine, _batch=False, _weights=None, alpha=None):
    """Open-frame result columns from expert scores alone (no model or feature access)."""
    if not _batch and 'drift' in engine:
        return analyze_batch(scores, engine, alpha)[0]
    fused = fuse_open(scores, engine['fusion'])
    share = novelty_share(fused['ds_bel_unknown'], fused['ds_bel_normal'])
    if _weights is None:
        sets = predict_conformal(engine['conformal'], fused['ds_betp'], betp_normal=fused['ds_betp_normal'], novelty_share=share)
    else:
        sets = weighted_sets(_weights['table'], engine['conformal']['epsilon'], fused['ds_betp'], share, _weights['test'])
    low, high = predict_va(engine['venn_abers'], fused['ds_betp'])
    out = {**fused, **sets, 'va_low': low, 'va_high': high, 'typ_score': np.asarray(scores['typ'], dtype=np.float64)}
    gate = conflict_gate(engine, fused['ds_conflict'])
    out['engine_review'] = (sets['cp_set'] == UNSURE) | gate
    if 'conflict_gate' in engine:
        out['ow_conflict_gate'] = gate
    if engine.get('states') == STATE_RULES:
        out.update(four_states(out, gate, engine['conformal']['epsilon']))
        out['engine_review'] = out['ow_queue'] != ''
    return out


def _canonical(value, digest):
    """Feed a stable, type-tagged encoding of a parameter tree into digest."""
    if isinstance(value, dict):
        digest.update(b'{')
        for key in sorted(value, key=str):
            if str(key).startswith('_'):
                continue  # runtime-only annotations such as _artifact_identity
            digest.update(json.dumps(str(key)).encode()); digest.update(b':')
            _canonical(value[key], digest)
        digest.update(b'}')
    elif isinstance(value, (list, tuple)):
        digest.update(b'[')
        for item in value:
            _canonical(item, digest)
        digest.update(b']')
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            _canonical(array.tolist(), digest)
        else:
            digest.update(('nd:%s:%s:' % (array.dtype.str, array.shape)).encode()); digest.update(array.tobytes())
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b'b1' if value else b'b0')
    elif isinstance(value, (int, np.integer)):
        digest.update(b'i' + str(int(value)).encode())
    elif isinstance(value, (float, np.floating)):
        digest.update(b'f' + float(value).hex().encode())
    elif isinstance(value, str):
        digest.update(b's' + json.dumps(value).encode())
    elif value is None:
        digest.update(b'n')
    else:
        raise TypeError('engine parameter of unsupported type: ' + type(value).__name__)
    digest.update(b';')


def engine_version(engine):
    """Format 1 keeps the frozen 1.3.7 identity; format 2 hashes every stored parameter (P1-7)."""
    if engine.get('format', 1) == 1:
        digest = hashlib.sha256(repr((engine['model_version'], engine['fusion']['reliability'],
                                      engine['conflict_cut'], engine['conformal']['epsilon'])).encode())
        return 'engine-' + digest.hexdigest()[:12]
    digest = hashlib.sha256(b'liushao-engine-format-2;')
    _canonical(engine, digest)
    return 'engine2-' + digest.hexdigest()[:12]


def analyze(pred, features, bundle, engine, alpha=None, *, return_used=False):
    """Append engine columns in input order and optionally return the effective batch engine."""
    if engine is None or engine['model_version'] != bundle['model_version']:
        raise ValueError('研判引擎与底座模型版本不一致')
    out = pred.copy()
    if not len(features):
        for col in COLUMNS + (OPEN_COLUMNS + _optional_columns(engine) if is_open(engine) else []):
            out[col] = pd.Series(dtype=object if col in ('cp_set', 'engine_version') else
                                 bool if col in ('drift_flag', 'ow_conflict_gate') else
                                 object if col in ('ow_state', 'ow_queue', 'ow_normal_reject', 'guarantee_status', *B2_COLUMNS, *B3_COLUMNS) else
                                 'int64' if col == 'engine_rank' else 'Int64' if col in ('ow_event', 'novelty_cluster') else float)
        return (out, engine) if return_used else out
    if is_open(engine):
        # Typicality and the ECOD evidence column aggregate the same empirical tails.
        ecod = engine['ecod']
        left, right = tail_logs(ecod, features)
        typ = _score_typicality_from_tails(engine['typicality'], left, right, ecod['skew_sign'])
        ecod_score = _sum_tail_logs(left, right, ecod['skew_sign'])
        del left, right
        scores = validated_scores({'rf': pred.rf_score.to_numpy(dtype=np.float64),
                                   'lr': pred.lr_score.to_numpy(dtype=np.float64), 'typ': typ}, OPEN_SOURCES)
        if 'drift' in engine:
            values, used = analyze_batch(scores, engine, alpha)
        else:
            values, used = analyze_scores(scores, engine, alpha=alpha), engine
        if 'events' in engine:
            values.update(event_columns(event_ids(engine, features, values['ow_state'], values['engine_rank']), values['ow_state']))
        # ECOD's summed tail score is kept for the per-field decomposition shown in the evidence
        # panel; on the open frame it is not an evidence source (the typicality expert is).
        values['ecod_score'] = ecod_score
        values['engine_version'] = engine_version(engine)
        for col in COLUMNS + OPEN_COLUMNS + _optional_columns(engine):
            out[col] = values[col]
        return (out, used) if return_used else out
    scores = {'rf': pred.rf_score.to_numpy(), 'lr': pred.lr_score.to_numpy(),
              'ecod': score_ecod(engine['ecod'], features)}
    fused = fuse(scores, engine['fusion'])
    sets = predict_conformal(engine['conformal'], fused['ds_betp'])
    va_low, va_high = predict_va(engine['venn_abers'], fused['ds_betp'])
    out['ecod_score'] = scores['ecod']
    for col, value in {**fused, **sets}.items():
        out[col] = value
    out['va_low'], out['va_high'] = va_low, va_high
    out['engine_review'] = (out.cp_set == UNSURE) | (out.ds_conflict >= engine['conflict_cut'])
    out['engine_version'] = engine_version(engine)
    return (out, engine) if return_used else out


def has_engine(frame):
    return all(col in frame for col in COLUMNS)


def engine_ranked(frame):
    """Main queue: fused belief first; ties broken by BetP, then the frozen RF and LR scores (T14).

    The LR score is continuous, so records with equal belief and RF score no
    longer fall back to record_id order.
    """
    if not has_engine(frame):
        return ranked(frame)
    keys = ['ds_belief', 'ds_betp', 'rf_score', 'lr_score', 'record_id']
    keys = [key for key in keys if key in frame]
    return frame.sort_values(keys, ascending=[key == 'record_id' for key in keys], kind='stable')


def engine_review_ranked(frame):
    """Review queue: conformal "uncertain" or high-conflict records.

    Ordered by pignistic BetP closest to 0.5, then by conflict K (rule chosen on validation).
    """
    if not has_engine(frame):
        return review_ranked(frame)
    subset = frame[frame.engine_review].copy()
    subset['threshold_distance'] = (subset.rf_score - subset.rf_threshold).abs()
    subset['betp_distance'] = (subset.ds_betp - .5).abs()
    return subset.sort_values(['betp_distance', 'ds_conflict', 'record_id'], ascending=[True, False, True],
                              kind='stable')


@lru_cache(maxsize=2)
def _paths(rf, features):
    return tree_paths(rf, list(features))


def explain(features_row, bundle, top_k=8):
    """Top-k tree-path contributions for one record (frame with a single row)."""
    rf, names = bundle['rf'], tuple(bundle['config']['features'])
    paths = _paths(rf, names)
    contrib, bias = contributions(paths, rf, features_row[list(names)])
    row = contrib[0]
    order = np.argsort(-np.abs(row))[:top_k]
    table = pd.DataFrame({'字段': [names[i] for i in order], '贡献': row[order]})
    table['方向'] = np.where(table['贡献'] >= 0, '推高攻击得分', '拉低攻击得分')
    return table, float(bias), float(bias + row.sum())


OPEN_SOURCE_LABELS = {'rf': ('随机森林 RF', '已知攻击 {A}', '非已知攻击 {N,U}'),
                      'lr': ('逻辑回归 LR', '已知攻击 {A}', '非已知攻击 {N,U}'),
                      'typ': ('典型性（ECOD 尾部前 3 维）', '非典型 {A,U}', '典型 {N,A}')}


def _open_source_masses(record, engine):
    fusion, rows = engine['fusion'], []
    for name in fusion['sources']:
        label, first, second = OPEN_SOURCE_LABELS[name]
        typ = fusion['kinds'][name] == 'typicality'
        if typ and fusion['atypical_focal'] == 'U':
            first = '非典型 {U}'
        score = float(record['typ_score' if typ else name + '_score'])
        m = source_mass(name, np.array([score]), fusion)[0]
        focal = (ATYPICAL_FOCAL[fusion['atypical_focal']], FOCAL_N | FOCAL_A) if typ else (FOCAL_A, FOCAL_N | FOCAL_U)
        rows.append({'证据源': label, '原始分数': score, '可靠度 r': fusion['reliability'][name],
                     '主张一': first, 'm(主张一)': float(m[focal[0]]), '主张二': second, 'm(主张二)': float(m[focal[1]]),
                     'm(未知Θ)': float(m[fusion['theta']])})
    return pd.DataFrame(rows)


def source_masses(record, engine):
    """Per-source basic probability assignment for one prediction row (a Series)."""
    if is_open(engine):
        return _open_source_masses(record, engine)
    rows = []
    for name, label in (('rf', '随机森林 RF'), ('lr', '逻辑回归 LR'), ('ecod', '无监督 ECOD')):
        score = record['ecod_score' if name == 'ecod' else name + '_score']
        p = calibrate(engine['fusion']['calibrators'][name], [score])
        m = bpa(p, engine['fusion']['reliability'][name])[0]
        rows.append({'证据源': label, '原始分数': float(score), '可靠度 r': engine['fusion']['reliability'][name],
                     'm(攻击)': m[0], 'm(正常)': m[1], 'm(未知Θ)': m[2]})
    return pd.DataFrame(rows)


def ecod_explain(features_row, engine, top_k=6):
    """Features with the largest ECOD tail contribution -log F for one record."""
    from .ecod import tail_logs
    left, right = tail_logs(engine['ecod'], features_row.to_numpy(dtype=float))
    tail = np.maximum(left[0], right[0])
    order = np.argsort(-tail)[:top_k]
    side = np.where(left[0] >= right[0], '偏低（左尾）', '偏高（右尾）')
    return pd.DataFrame({'字段': [features_row.columns[i] for i in order], '尾部贡献': tail[order],
                         '方向': side[order]})


def drift_space(bundle, x):
    """Imputed, signed-log, standardised space of the frozen LR pipeline."""
    return np.asarray(bundle['lr'][:-1].transform(x), dtype=np.float32)


def drift_check(engine, bundle, features, max_rows=1000, n_perm=200):
    return mmd_test(engine['drift_reference'], drift_space(bundle, features), n_perm=n_perm, max_rows=max_rows)


def load_engine(path, bundle):
    """Load artifacts/engine.joblib (format 1 or 2); None when absent or bound to another base model."""
    path = Path(path)
    if not path.exists():
        return None
    engine = joblib.load(path)
    if engine.get('format') not in SUPPORTED_FORMATS or engine.get('model_version') != bundle['model_version']:
        return None
    validate_engine_compatibility(engine)
    return engine


__all__ = ['ATTACK', 'NORMAL', 'UNSURE', 'COLUMNS', 'DEFAULT_ALPHA', 'EVENT_COLUMNS', 'RISK_ALPHAS', 'RISK_COLUMNS', 'alpha_for_budget', 'apply_alpha', 'crc_threshold', 'event_ids', 'risk_table', 'STATES', 'STATE_COLUMNS', 'STATE_RULES', 'four_states', 'review_rank', 'batch_weights', 'density_ratio_weights', 'ess', 'CONFLICT_GATES', 'conflict_gate', 'DISCOUNT_FORMS', 'DRIFT_COLUMNS', 'analyze_batch', 'batch_engine', 'with_discount', 'ENGINE_FORMAT', 'OPEN_COLUMNS', 'OPEN_WORLD_FIELDS', 'SUPPORTED_FORMATS',
           'analyze', 'analyze_scores', 'drift_check', 'engine_ranked', 'engine_review_ranked', 'engine_version', 'expert_scores',
           'explain', 'fit_engine', 'half_b_thirds', 'has_engine', 'is_open', 'load_engine', 'source_masses']
