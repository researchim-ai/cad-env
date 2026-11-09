import hashlib

import pytest

from cad_env.drawings import font_cache


def test_install_is_offline_after_verified_cache_and_repairs_corruption(
    tmp_path, monkeypatch
):
    data = b"test font bytes"
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        font_cache, "ASSETS", {"osifont.ttf": ("https://example.invalid/font", sha)}
    )
    calls = []

    def fetch(url):
        calls.append(url)
        return data

    folder = font_cache.install(tmp_path, fetch)
    font_cache.install(folder, fetch)
    assert len(calls) == 1
    (folder / "osifont.ttf").write_bytes(b"corrupted")
    font_cache.install(folder, fetch)
    assert len(calls) == 2
    assert (folder / "osifont.ttf").read_bytes() == data


def test_mismatched_download_never_replaces_existing_file(tmp_path, monkeypatch):
    (tmp_path / "osifont.ttf").write_bytes(b"existing")
    monkeypatch.setattr(
        font_cache,
        "ASSETS",
        {"osifont.ttf": ("https://example.invalid/font", "0" * 64)},
    )
    with pytest.raises(ValueError, match="checksum"):
        font_cache.install(tmp_path, lambda _: b"wrong download")
    assert (tmp_path / "osifont.ttf").read_bytes() == b"existing"


def test_missing_font_requires_explicit_install_and_corruption_is_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CAD_ENV_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="--install"):
        font_cache.require_font()
    path = font_cache.font_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong font")
    with pytest.raises(ValueError, match="checksum"):
        font_cache.require_font()
