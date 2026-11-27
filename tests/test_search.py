"""Synthetic fixtures verify Search mechanics. V1 effectiveness requires real data."""

from dataclasses import replace

import pandas as pd
import pytest

from app_engine_evidence import IDENTITY_COLUMNS, save_event_reviews
from liushao.reviews import current_reviews, save_review
from liushao.search import (
    ANOMALY, NORMAL, AnalysisContext, PriorCheck, SearchScope, search_counterexamples,
)


CONTEXT = AnalysisContext('a' * 64, 'test-model', 'b' * 64, 'c' * 64, ('d' * 64, 'e' * 64, 'v1'), .05)
ENGINE = dict(zip(IDENTITY_COLUMNS, CONTEXT.engine_identity))


def row(rid, physical, **columns):
    return dict(record_id=rid, physical=physical, source_file='source.csv', source_sha256='f' * 64,
                source_row=physical + 100, model_version=CONTEXT.model_version, **ENGINE,
                **{'risk_alpha': .05, 'ow_event': 0, 'ow_queue': '模糊', 'ow_state': '正常',
                   'rf_anomaly': False, 'lr_anomaly': False, 'near_threshold': False,
                   'cp_set': '正常', 'cp_set_size': 1, 'ds_conflict': 0.,
                   'ow_conflict_gate': False, 'va_low': .1, 'va_high': .2, **columns})


def run(rows, scope=None, *, context=CONTEXT, **options):
    frame = pd.DataFrame(rows)
    physical = frame.pop('physical')
    return search_counterexamples(frame, physical, context=context,
                                  scope=scope or SearchScope(0, NORMAL), **options)


def snapshot(record):
    return dict(schema_version=2, method_version='fixture', record_id=record['record_id'],
                model_version=CONTEXT.model_version, input_sha256=CONTEXT.input_sha256,
                model_sha256=CONTEXT.model_sha256, descriptor_sha256=CONTEXT.descriptor_sha256,
                input_csv_row=record['physical'],
                identity={k: record[k] for k in ('source_file', 'source_sha256', 'source_row')},
                scores=dict(rf=.1, lr=.2, rf_threshold=.5, lr_threshold=.5), cards=[],
                selection={}, decomposition={},
                engine=dict(status='verified', identity=ENGINE, open_world=dict(alpha=.05)))


def save_direct(path, record, status, *, payload=True):
    save_review(path, record['record_id'], CONTEXT.model_version, status, '人工核查依据',
                evidence_snapshot=snapshot(record) if payload else None)


def by_id(result):
    return {item.record_id: item for item in result.ordered_candidates}


def sources(candidate):
    return {item.source: item for item in candidate.basis.evidence}


def test_synthetic_counterexamples_cover_all_tiers_and_four_evidence_types(tmp_path):
    rep, human = row('rep', 2), row('human', 3)
    rows = [rep, human, row('opposite', 4, ow_state='攻击'), row('conflict', 5, ow_conflict_gate=True, ds_conflict=.4),
            row('uncertain', 6, near_threshold=True), row('heterogeneous', 7, rf_anomaly=True, lr_anomaly=True),
            row('blind_spot', 8)]
    scope = SearchScope(0, NORMAL, frozenset({'rep'}), frozenset({'human', 'opposite', 'blind_spot'}))
    db = tmp_path / 'audit.db'
    save_direct(db, rep, NORMAL)
    save_direct(db, human, ANOMALY)
    result = run(rows, scope, reviews=current_reviews(db, CONTEXT.model_version), budget=2)
    ledger, = result.confirmed_counterexamples
    assert ledger.record_id == 'human'
    assert ledger.review_id == 2
    assert ledger.reason == '人工核查依据'
    assert ledger.context == CONTEXT
    assert ledger.counterexample_applies_to_authorization
    assert [item.record_id for item in result.supporting_reviews] == ['rep']
    candidates = by_id(result)
    assert {rid: item.priority for rid, item in candidates.items()} == {
        'opposite': 1, 'conflict': 2, 'uncertain': 2, 'heterogeneous': 2, 'blind_spot': 3,
    }
    assert sources(candidates['conflict'])['ow_conflict_gate'].category == 'model_conflict'
    assert sources(candidates['conflict'])['ds_conflict'].value == .4
    assert sources(candidates['uncertain'])['near_threshold'].category == 'prediction_uncertainty'
    assert sources(candidates['heterogeneous'])['member_comparison'].category == 'event_heterogeneity'
    assert [item.record_id for item in result.recommendations] == ['opposite', 'heterogeneous']
    assert all(item.reasons for item in result.ordered_candidates)


def test_queue_outside_the_event_and_deduplication():
    rows = [row('in_event', 2), row('flagged', 3, ow_event=None, ow_queue='', lr_anomaly=True),
            row('both', 4, ow_queue=''), row('not_pending', 5, ow_event=None, ow_queue=''),
            row('queued', 6, ow_event=None)]
    scope = SearchScope(0, NORMAL, pending_release_ids=frozenset({'flagged', 'both', 'queued'}))
    result = run(rows, scope)
    assert [c.record_id for c in result.recommendations] == ['flagged', 'in_event', 'both']
    flagged = result.recommendations[0]
    assert (flagged.paths, flagged.native_queue, flagged.priority) == (('release',), '', 1)
    assert 'LR' in flagged.reasons[0]
    both = result.recommendations[-1]
    assert both.paths == ('event', 'release')
    assert len(both.reasons) == 3


def test_ties_prefer_uncovered_then_native_rank(tmp_path):
    rep = row('rep', 2, engine_rank=1)
    rows = [rep, row('covered_first', 3, engine_rank=2, near_threshold=True),
            row('uncovered_later', 4, engine_rank=4, near_threshold=True, rf_anomaly=True),
            row('uncovered_earlier', 5, engine_rank=3, near_threshold=True, rf_anomaly=True)]
    db = tmp_path / 'audit.db'
    save_direct(db, rep, NORMAL)
    scope = SearchScope(0, NORMAL, frozenset({'rep'}))
    result = run(rows, scope, reviews=current_reviews(db, CONTEXT.model_version))
    assert result.engine_rank_used
    assert [c.record_id for c in result.recommendations] == [
        'uncovered_earlier', 'uncovered_later', 'covered_first',
    ]
    assert all(c.priority == 2 for c in result.recommendations)


@pytest.mark.parametrize('ranks', [(None, 2, 1), (2, 2, 1), (False, 2, 1)])
def test_incomparable_rank_uses_physical_then_record_identity(ranks):
    rows = [row('z', 2, engine_rank=ranks[0]), row('a', 2, engine_rank=ranks[1]),
            row('first', 9, engine_rank=ranks[2])]
    result = run(rows)
    assert not result.engine_rank_used
    assert [c.record_id for c in result.recommendations] == ['a', 'z', 'first']
    assert list(by_id(run(rows[::-1]))) == list(by_id(result))


def test_confirmed_review_is_kept_out_of_candidate_budget_even_if_snapshot_unavailable(tmp_path):
    rows = [row('legacy', 2), row('mismatched', 3), row('invalid', 4), row('still_new', 5)]
    db = tmp_path / 'audit.db'
    save_direct(db, rows[0], ANOMALY, payload=False)
    save_direct(db, rows[1], ANOMALY)
    save_direct(db, rows[2], ANOMALY)
    reviews = current_reviews(db, CONTEXT.model_version)
    reviews['invalid']['evidence_sha256'] = 'bad-digest'
    rows[1]['source_row'] += 1
    result = run(rows, reviews=reviews, budget=1)
    assert not result.confirmed_counterexamples
    assert [c.record_id for c in result.pending_verification] == ['legacy', 'mismatched', 'invalid']
    assert list(by_id(result)) == ['still_new']


def test_direct_representative_disagreement_and_inheritance_provenance(tmp_path):
    good, bad, inherited = row('good', 2), row('bad', 3), row('inherited', 4)
    db = tmp_path / 'audit.db'
    save_direct(db, good, NORMAL)
    save_direct(db, bad, ANOMALY)
    payload = snapshot(inherited)
    payload.pop('engine')
    payload['event_review'] = dict(engine_identity=ENGINE, open_world=dict(alpha=.05))
    save_event_reviews(db, CONTEXT.model_version, ANOMALY, '继承依据', {'inherited': 0}, [payload])
    scope = SearchScope(0, NORMAL, frozenset({'good', 'bad'}), frozenset({'inherited'}))
    result = run([good, bad, inherited], scope, reviews=current_reviews(db, CONTEXT.model_version))
    assert [r.record_id for r in result.supporting_reviews] == ['good']
    contradiction, = result.confirmed_counterexamples
    assert contradiction.record_id == 'bad'
    assert contradiction.counterexample_applies_to_authorization
    assert sources(by_id(result)['inherited'])['event_review'].state == 'available'


def test_completed_check_reopens_only_with_changed_relevant_basis_and_insufficiency_reason():
    rows = [row('old', 2), row('new', 3)]
    first = run(rows)
    prior = {'old': PriorCheck(by_id(first)['old'].basis, 'checked', '已核查日志')}
    assert list(by_id(run(rows, prior_checks=prior, budget=0))) == ['new']
    same = run(rows, prior_checks=prior, reopen={'old': '尝试重派'})
    assert [c.record_id for c in same.retained_checks] == ['old']
    rows[0]['va_high'] = .3
    assert list(by_id(run(rows, prior_checks=prior))) == ['new']
    reopened = run(rows, prior_checks=prior, reopen={'old': '端点变更，需要重新核对旧处置依据'})
    assert '端点变更，需要重新核对旧处置依据' in by_id(reopened)['old'].reasons
    assert reopened.retained_checks == ()


def test_checked_form_covers_only_matching_verified_evidence():
    rep = row('rep', 2)
    checked = row('checked', 3, rf_anomaly=True)
    other = row('similar', 4, rf_anomaly=True)
    scope = SearchScope(0, NORMAL, frozenset({'rep'}))
    first = run([rep, checked, other], scope)
    assert by_id(first)['similar'].uncovered_form
    prior = {'checked': PriorCheck(by_id(first)['checked'].basis, 'checked', '已核查该形态')}
    result = run([rep, checked, other], scope, prior_checks=prior)
    assert not by_id(result)['similar'].uncovered_form
    assert 'representative_comparison' in sources(by_id(result)['similar'])
    checked['va_high'] = .3
    assert by_id(run([rep, checked, other], scope, prior_checks=prior))['similar'].uncovered_form


def test_pending_evidence_keeps_original_task_and_supports_explicit_resume(tmp_path):
    record = row('pending', 2)
    db = tmp_path / 'audit.db'
    save_direct(db, record, '待补证', payload=False)
    options = {'reviews': current_reviews(db, CONTEXT.model_version)}
    first = run([record], **options)
    pending, = first.pending_verification
    prior = {'pending': PriorCheck(pending.basis, 'pending_evidence', '等资产日志')}
    again = run([record], prior_checks=prior, **options)
    assert again.ordered_candidates == ()
    assert again.pending_verification[0].reasons == ('等资产日志',)
    resumed = run([record], prior_checks=prior, reopen={'pending': '人工续办原任务'}, **options)
    assert resumed.recommendations[0].prior_check == prior['pending']


def test_missing_and_inapplicable_evidence_do_not_create_a_signal_or_fill_zero():
    bare = row('bare', 2)
    for key in (*IDENTITY_COLUMNS, 'risk_alpha', 'cp_set', 'ow_state'):
        bare.pop(key)
    bare['near_threshold'] = False
    bare['disagreement'] = False
    context = replace(CONTEXT, engine_identity=(), risk_alpha=None)
    item, = run([bare], context=context).recommendations
    assert item.priority == 3
    assert sources(item)['ds_conflict'].state == 'not_applicable'
    assert sources(item)['ds_conflict'].value is None
    derived = run([row('missing', 2, va_low=float('nan'), guarantee_status='失效', drift_flag=False,
                       drift_delta=float('nan'))],
                  required={'missing': {'guarantee_status'}})
    critical, = derived.recommendations
    assert critical.priority == 1
    assert sources(critical)['venn_abers'].state == 'unknown'
    assert ('drift', 'not_tested') in {(f.source, f.state) for f in derived.batch_diagnostics}
    assert ('guarantee_status', 'invalid') in {(f.source, f.state) for f in derived.batch_diagnostics}


def test_direction_of_model_conflict_and_conformal_empty_vs_two_labels():
    rows = [row('empty', 2, cp_set='不确定', cp_set_size=0, ow_state='新异'),
            row('double', 3, cp_set='不确定', cp_set_size=2, ow_state='模糊', lr_anomaly=True),
            row('interval', 4, va_low=0., va_high=1.)]
    first, second, third = run(rows).recommendations
    assert '空集' in sources(first)['cp_set'].reason
    assert '双标签' in sources(second)['cp_set'].reason
    assert sources(second)['RF/LR'].value == (NORMAL, ANOMALY)
    assert third.priority == 3


def test_ecod_shape_and_training_reference_require_current_bound_record_evidence():
    rep, member = row('rep', 2), row('member', 3)
    scope = SearchScope(0, NORMAL, frozenset({'rep'}))
    rep_evidence, member_evidence = snapshot(rep), snapshot(member)
    for payload, side in ((rep_evidence, 'left'), (member_evidence, 'right')):
        payload['engine']['ecod_decomposition'] = {
            'top_fields': [{'field': 'Duration', 'direction': side, 'input_missing': False}],
        }
        payload['cards'] = [{'field': 'Duration', 'missing': False, 'role': '反向线索',
                             'reference_position': {'valid': True, 'status': '超出训练正常参考范围'}}]
    evidence = {'rep': rep_evidence, 'member': member_evidence}
    item = by_id(run([rep, member], scope, evidence=evidence))['member']
    assert item.priority == 2
    assert item.uncovered_form
    assert sources(item)['training_reference'].value[0][0] == 'Duration'
    assert sources(item)['reference_replacement'].value == (('Duration', '反向线索'),)
    member_evidence['input_sha256'] = 'wrong-context'
    stale = by_id(run([rep, member], scope, evidence=evidence))['member']
    assert stale.priority == 3
    assert sources(stale)['ecod_tails'].state == 'unknown'


def test_identity_mismatch_routes_to_verification_and_disables_cross_context_comparison():
    rep = row('other-analysis', 2, risk_alpha=.1, rf_anomaly=True, engine_rank=1)
    current = row('current', 3, engine_rank=2)
    scope = SearchScope(0, NORMAL, frozenset({'other-analysis'}))
    result = run([rep, current], scope)
    assert [c.record_id for c in result.pending_verification] == ['other-analysis']
    only, = result.recommendations
    assert only.priority == 3
    assert not only.uncovered_form
    assert sources(only)['member_comparison'].state == 'unknown'
