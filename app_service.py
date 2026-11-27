"""Atomic input analysis, separate from Streamlit rendering."""
import hashlib
import io
import json
import time
from threading import RLock

import pandas as pd

from app_engine_trust import MAX_DESCRIPTOR_BYTES, MAX_ENGINE_BYTES, matches_release, read_bounded
from app_quality import build_quality_view, validate_final_mapping
from app_types import ActiveInput, AnalysisResult, EngineStatus, ReviewSnapshot, validate_history
from delivery.safe_files import read_json
from liushao import reviews as review_store
from liushao.data import read_csv_checked
from liushao.model import infer

MAX_BYTES = 50 * 1024 * 1024
MAX_ROWS = 50000
_ENGINE_CACHE = {}
_ENGINE_CACHE_LOCK = RLock()


ENGINE_DEGRADED = {
    'missing': '研判引擎文件缺失（engine.joblib 或 engine.json），本次只显示 RF/LR 基础检测结果。',
    'tampered': '研判引擎文件未通过完整性校验（哈希不符或文件损坏），已停用引擎，本次只显示 RF/LR 基础检测结果。',
    'version_mismatch': '研判引擎与当前底座模型或程序版本不匹配，已停用引擎，本次只显示 RF/LR 基础检测结果。',
    'runtime_error': '研判引擎运行出错，已停用引擎，本次只显示 RF/LR 基础检测结果。',
}


def _degraded(code):
    return EngineStatus(code, ENGINE_DEGRADED[code])


def load_trusted_engine_status(artifacts, bundle):
    """Load the exact verified bytes; returns (engine or None, EngineStatus).

    Identity includes parameter AND descriptor hash.  Application fallback does
    not bypass the standard launcher's whole-package check.
    """
    try:
        import liushao_engine
        path, descriptor = artifacts / 'engine.joblib', artifacts / 'engine.json'
        if not path.is_file() or not descriptor.is_file():
            return None, _degraded('missing')
        try:
            descriptor_bytes = read_bounded(descriptor, MAX_DESCRIPTOR_BYTES)
            meta = read_json(descriptor_bytes)
            content = read_bounded(path, MAX_ENGINE_BYTES)
        except (OSError, ValueError, RecursionError):
            return None, _degraded('tampered')
        digest = hashlib.sha256(content).hexdigest()
        if not isinstance(meta, dict) or meta.get('engine_sha256') != digest:
            return None, _degraded('tampered')
        # A self-consistent descriptor cannot authorize arbitrary pickle bytes.
        # Validate all release identity fields before any joblib deserialization,
        # including cache hits, so incorrect provenance never reaches evidence.
        if (not matches_release(digest, meta, bundle)
                or meta['format'] not in liushao_engine.SUPPORTED_FORMATS):
            return None, _degraded('version_mismatch')
        descriptor_digest = hashlib.sha256(descriptor_bytes).hexdigest()
        key = (digest, descriptor_digest, bundle['model_version'])
        # Streamlit sessions share this cache.  Serialize misses and replacement,
        # including the lookup, so a concurrent descriptor cannot evict our key
        # between population and return.  Bytes are still verified on every call.
        with _ENGINE_CACHE_LOCK:
            if key not in _ENGINE_CACHE:
                import joblib
                try:
                    engine = joblib.load(io.BytesIO(content))
                except Exception:
                    return None, _degraded('tampered')
                if (not isinstance(engine, dict) or type(engine.get('format')) is not int
                        or engine['format'] != meta['format']
                        or engine.get('model_version') != bundle['model_version']
                        or liushao_engine.engine_version(engine) != meta['engine_version']):
                    return None, _degraded('version_mismatch')
                try:
                    liushao_engine.validate_engine_compatibility(engine)
                except ValueError as error:
                    return None, EngineStatus('version_mismatch', str(error))
                engine = dict(engine)
                from app_engine_inference import inference_version
                engine['_artifact_identity'] = {
                    'engine_sha256': digest, 'engine_descriptor_sha256': descriptor_digest,
                    'engine_version': meta['engine_version'], 'base_model_version': bundle['model_version'],
                    'base_model_sha256': meta['base_model_sha256'],
                    'engine_inference_version': inference_version(engine)}
                _ENGINE_CACHE.clear()
                _ENGINE_CACHE[key] = engine
            return _ENGINE_CACHE[key], EngineStatus('ok', '')
    except Exception:
        return None, _degraded('runtime_error')


def load_trusted_engine(artifacts, bundle):
    return load_trusted_engine_status(artifacts, bundle)[0]


def with_engine_status(pred, features, bundle, artifacts):
    """Append trusted-triage columns; any failure keeps the frozen 1.3.6 predictions and says why."""
    engine, status = load_trusted_engine_status(artifacts, bundle)
    if engine is None:
        return pred, status
    try:
        from app_engine_inference import analyze_current_engine
        result = analyze_current_engine(pred, features, bundle, engine)
        from app_engine_evidence import IDENTITY_COLUMNS
        for name in IDENTITY_COLUMNS:
            result[name] = engine['_artifact_identity'][name]
        return result, status
    except Exception:
        return pred, _degraded('runtime_error')


def with_engine(pred, features, bundle, artifacts):
    return with_engine_status(pred, features, bundle, artifacts)[0]


def analyze_input(content, name, source, bundle, model_identity, artifacts, *, preflight=None):
    tick = time.perf_counter()
    if len(content) > MAX_BYTES:
        raise ValueError('文件超过 50 MiB，请分成较小的文件后再上传。')
    digest = hashlib.sha256(content).hexdigest()
    raw = preflight.take(content, model_identity) if preflight is not None else None
    if raw is None:
        raw = read_csv_checked(io.BytesIO(content), max_rows=MAX_ROWS)
    if raw.empty or len(raw) > MAX_ROWS:
        raise ValueError('文件需要包含 1 至 50,000 条记录。')
    pred, features, rejected = infer(raw, bundle, digest, name)
    # The trial ships the sample CSV alone. Its physical rows are its provenance.
    # Timestamps are metadata. Preserve the frozen feature adapter and model.
    from app_input import input_times
    if 'Timestamp' in raw:
        stamps, basis = input_times(raw.Timestamp)
        by_line = dict(zip(raw.attrs['source_rows'], zip(stamps, basis)))
        for part in (pred, rejected):
            values = [by_line[line] for line in part.source_row]
            part['timestamp'] = pd.to_datetime([v[0] for v in values])
            part['timestamp_basis'] = [v[1] for v in values]
    pred, engine_status = with_engine_status(pred, features, bundle, artifacts)
    physical = pred.source_row.copy()
    original = pred
    try:
        quality = build_quality_view(raw, original, features, rejected, final_predictions=pred)
        error = None
    except Exception as exc:
        quality, error = None, str(exc)
    from app_engine_evidence import engine_identity
    engine_key = engine_identity(pred.iloc[0]) if len(pred) else ('', '')
    active = ActiveInput(content, name, source, digest,
                         '|'.join((digest, source, name, model_identity, *engine_key)), model_identity)
    return active, AnalysisResult(pred, features, rejected, time.perf_counter()-tick, len(raw), engine_status), physical, quality, error



def read_review_snapshot(path, model_version, record_ids):
    """One live query; release unrelated history before returning to rendering."""
    rows = validate_history(review_store.history(path, model_version=model_version, record_ids=record_ids, include_evidence=False), model_version)
    audit = [item for item in rows if item['record_id'] in record_ids]
    from app_record_tools import latest_reviews
    return ReviewSnapshot(latest_reviews(audit, model_version), audit)