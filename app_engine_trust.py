"""Release-owned engine identities checked before deserializing joblib bytes.

An engine descriptor is metadata, not a trust anchor. Register a new frozen
engine only after reviewing its provenance; descriptor notes remain editable.
"""

# Covers both shipped compressed artifacts and their ~15 MB uncompressed form.
MAX_ENGINE_BYTES = 16 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 256 * 1024

_BASE = {
    'base_model_version': 'c7a028b4e4cafedd',
    'base_model_sha256': '3c5b425f3a34454bd2b5a2738535742cc03e0f8f6e55fefc5abc5929c4dccad7',
    'data_version': 'e946961bf545243c081c8995ae582b56c968bf35094d12c2e0a17d7bfea04b87',
}
TRUSTED_ENGINES = {
    'adef9cd8ae094557810ba924340e7f7d6869d7a789749f4ab9f006d4bbf39838':
        dict(_BASE, format=2, engine_version='engine2-a6235b8cbf97'),
    '01fe41d5349d7161237f981c29b3c61db40a8742a6f157f79442c6dd1014c131':
        dict(_BASE, format=1, engine_version='engine-00dca9fe9e73'),
}


def read_bounded(path, limit):
    """Bound the actual read rather than relying on a racy size check."""
    with path.open('rb') as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise ValueError('Engine artifact exceeds the supported size')
    return content


def matches_release(digest, metadata, bundle):
    expected = TRUSTED_ENGINES.get(digest)
    return (expected is not None
            and all(type(metadata.get(name)) is type(value) and metadata[name] == value
                    for name, value in expected.items())
            and bundle.get('model_version') == expected['base_model_version']
            and bundle.get('model_sha256') == expected['base_model_sha256']
            and bundle.get('config', {}).get('data_version') == expected['data_version'])
