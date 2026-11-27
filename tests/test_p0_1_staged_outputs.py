import errno
import json
from pathlib import Path
import sys

import pytest

from delivery.output_directory import staged_output_directory


def test_interrupted_study_write_leaves_no_final_directory_and_can_retry(tmp_path, monkeypatch):
    study = pytest.importorskip('scripts.analyze_review_study', reason='试用包不包含研究汇总工具')
    source = tmp_path / 'input.json'
    source.write_bytes(b'[]')
    destination = tmp_path / 'study'
    monkeypatch.setattr(study, 'ROOT', tmp_path / 'checkout')
    monkeypatch.setattr(study, 'analyze', lambda *_: {'complete': True})
    monkeypatch.setattr(sys, 'argv', ['study', '--session', str(source), '--history', str(source),
                                     '--output', str(destination)])
    original_write = Path.write_bytes

    def partial_write(path, content):
        assert path.parent.parent == tmp_path and path.parent != destination
        assert not destination.exists()
        original_write(path, content[:8])
        raise OSError(errno.ENOSPC, 'synthetic disk failure')

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'write_bytes', partial_write)
        with pytest.raises(OSError, match='synthetic disk failure'):
            study.main()
    assert set(tmp_path.iterdir()) == {source}
    study.main()
    summary = destination / 'study_summary.json'
    complete = summary.read_bytes()
    assert json.loads(complete)['complete'] is True and source.read_bytes() == b'[]'
    with pytest.raises(FileExistsError):
        study.main()
    assert summary.read_bytes() == complete


def test_interrupted_material_generation_leaves_no_final_directory_and_can_retry(tmp_path, monkeypatch):
    materials = pytest.importorskip('scripts.build_t16_materials', reason='试用包不包含 T16 制图工具')
    archive = tmp_path / 'archive'
    archive.mkdir()
    source = archive / 'frozen-result.json'
    source.write_bytes(b'{}')
    destination = tmp_path / 'materials'

    def fail(input_path, staging):
        assert input_path == archive
        assert staging.parent == destination.parent and staging != destination
        (staging / 'partial').write_bytes(b'{')
        assert not destination.exists()
        raise OSError(errno.ENOSPC, 'synthetic disk failure')

    with monkeypatch.context() as patch:
        patch.setattr(materials, '_build_into', fail)
        with pytest.raises(OSError, match='synthetic disk failure'):
            materials.build(archive, destination)
    assert set(tmp_path.iterdir()) == {archive} and source.read_bytes() == b'{}'

    def finish(input_path, staging):
        assert input_path == archive and not destination.exists()
        (staging / 'complete.json').write_bytes(b'{}')
        return {'figures': [], 'plotted_values': 0}

    monkeypatch.setattr(materials, '_build_into', finish)
    assert materials.build(archive, destination)['figures'] == []
    assert {item.name for item in destination.iterdir()} == {'complete.json'}
    with pytest.raises(FileExistsError):
        materials.build(archive, destination)
    assert (destination / 'complete.json').read_bytes() == b'{}'


@pytest.mark.parametrize('with_content', [False, True])
def test_publication_never_replaces_concurrent_destination(tmp_path, with_content):
    destination = tmp_path / 'output'
    with pytest.raises(FileExistsError):
        with staged_output_directory(destination) as stage:
            (stage / 'ours').write_bytes(b'ours')
            destination.mkdir()
            if with_content:
                (destination / 'theirs').write_bytes(b'theirs')
    assert not stage.exists()
    assert [item.name for item in destination.iterdir()] == (['theirs'] if with_content else [])
