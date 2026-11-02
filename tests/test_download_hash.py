import hashlib
import os

import pytest

from pyiqa.utils.download_util import _hash_prefix_from_name, load_file_from_url


@pytest.mark.parametrize(
    'name, expected',
    [
        ('tres_koniq-f0502926.pth', 'f0502926'),
        ('pt_inception-2015-12-05-6726825d.pth', '6726825d'),
        ('ViT-B-32.pt', None),
        ('brisque_svm_weights.pth', None),
        ('niqe_modelparameters.mat', None),
    ],
)
def test_hash_prefix_from_name(name, expected):
    assert _hash_prefix_from_name(name) == expected


def _serve(tmp_path, payload, prefix):
    src = tmp_path / 'src'
    src.mkdir()
    path = src / f'weights-{prefix}.pth'
    path.write_bytes(payload)
    return path.as_uri()


def test_download_with_matching_hash_is_cached(tmp_path):
    payload = b'pyiqa weights'
    url = _serve(tmp_path, payload, hashlib.sha256(payload).hexdigest()[:8])
    cache = tmp_path / 'cache'

    cached = load_file_from_url(url, model_dir=str(cache), progress=False)

    with open(cached, 'rb') as f:
        assert f.read() == payload


def test_download_with_wrong_hash_raises_and_is_not_cached(tmp_path):
    # Simulates a truncated or tampered download: the bytes do not match the
    # hash in the file name.
    url = _serve(tmp_path, b'truncated', 'deadbeef')
    cache = tmp_path / 'cache'

    with pytest.raises(RuntimeError, match='invalid hash value'):
        load_file_from_url(url, model_dir=str(cache), progress=False)

    # Nothing left behind that a later run would treat as a valid cache hit.
    assert not os.path.exists(cache / 'weights-deadbeef.pth')
