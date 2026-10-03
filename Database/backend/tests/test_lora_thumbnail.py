"""Existing sidecar thumbnails never expose arbitrary paths or mutate data."""
import hashlib
import io
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

import lora_api_server as api
from lora_thumbnail import ThumbnailError, read_thumbnail

PNG = b"\x89PNG\r\n\x1a\nfixture"
JPEG = b"\xff\xd8\xfffixture"
WEBP = b"RIFF\x04\x00\x00\x00WEBPfixture"


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    source = root / "portrait.safetensors"
    source.write_bytes(b"LoRA remains unread and unchanged")
    return root, source


@pytest.mark.parametrize("extension,content,mime", [
    (".jpeg", JPEG, "image/jpeg"), (".jpg", JPEG, "image/jpeg"),
    (".png", PNG, "image/png"), (".webp", WEBP, "image/webp"),
])
def test_supported_exact_basename_raster_is_returned_without_writes(files, extension, content, mime):
    root, source = files
    image = source.with_suffix(extension)
    image.write_bytes(content)
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in (source, image)]
    assert read_thumbnail(source, root) == (content, mime)
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in (source, image)]


def test_small_basename_image_preferred_to_preview_fallback(files):
    root, source = files
    preview = root / "portrait.preview.jpeg"
    preview.write_bytes(JPEG + b"preview")
    assert read_thumbnail(source, root)[0] == JPEG + b"preview"
    source.with_suffix(".jpeg").write_bytes(JPEG)
    assert read_thumbnail(source, root)[0] == JPEG


@pytest.mark.parametrize('content,mime', [(WEBP, 'image/webp'), (PNG, 'image/png')])
def test_downloaded_jpeg_filename_uses_actual_supported_raster_type(files, content, mime):
    root, source = files
    source.with_suffix('.jpeg').write_bytes(content)
    assert read_thumbnail(source, root) == (content, mime)


def test_only_exact_name_and_supported_extensions_are_considered(files):
    root, source = files
    for filename in ("portrait.svg", "portrait-other.png", "portrait.safetensors.png", "random.jpeg"):
        (root / filename).write_bytes(PNG)
    with pytest.raises(ThumbnailError) as error:
        read_thumbnail(source, root)
    assert error.value.status_code == 404


@pytest.mark.parametrize('extension', ['.png', '.jpeg'])
def test_svg_disguised_as_raster_is_rejected(files, extension):
    root, source = files
    source.with_suffix(extension).write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"/>')
    with pytest.raises(ThumbnailError) as error:
        read_thumbnail(source, root)
    assert error.value.status_code == 415


def test_catalogue_path_outside_root_or_missing_source_is_rejected(files, tmp_path):
    root, source = files
    outside = tmp_path / "secret.safetensors"
    outside.write_bytes(b"outside")
    outside.with_suffix(".png").write_bytes(PNG)
    for path in (outside, root / "missing.safetensors", None):
        with pytest.raises(ThumbnailError) as error:
            read_thumbnail(path, root)
        assert error.value.status_code == 404


def test_symlinked_sidecar_cannot_serve_another_file(files, tmp_path):
    root, source = files
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    try:
        source.with_suffix('.png').symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links; Linux CI exercises containment")
    with pytest.raises(ThumbnailError) as error:
        read_thumbnail(source, root)
    assert error.value.status_code == 404


def test_stat_size_limit_blocks_open_and_read_limit_blocks_growing_file(files, monkeypatch):
    root, source = files
    image = source.with_suffix('.png')
    image.write_bytes(PNG + b"oversized")
    with pytest.raises(ThumbnailError) as error:
        read_thumbnail(source, root, max_bytes=8)
    assert error.value.status_code == 413
    image.write_bytes(PNG)
    original_open = Path.open
    sizes = []
    class Growing(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)
    def opened(path, *args, **kwargs):
        if path == image:
            return Growing(PNG + b"growing beyond limit")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(ThumbnailError) as error:
        read_thumbnail(source, root, max_bytes=20)
    assert error.value.status_code == 413
    assert sizes == [21]


def test_http_uses_catalogue_identity_readonly_and_no_arbitrary_path(files, tmp_path, monkeypatch):
    root, source = files
    source.with_suffix('.png').write_bytes(PNG)
    database = tmp_path / "catalogue.db"
    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE lora(stable_id TEXT, file_path TEXT)")
    conn.execute("INSERT INTO lora VALUES(?, ?)", ('known', str(source)))
    conn.commit()
    conn.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.setattr(api, 'DB_PATH', database)
    monkeypatch.setenv('LORA_ROOT', str(root))
    # No application lifespan: the endpoint itself must not initialize tables.
    client = TestClient(api.app)
    result = client.get('/api/lora/known/thumbnail')
    assert result.status_code == 200
    assert result.content == PNG
    assert result.headers['content-type'] == 'image/png'
    assert result.headers['x-content-type-options'] == 'nosniff'
    assert result.headers['cache-control'] == 'private, max-age=300'
    assert client.get('/api/lora/missing/thumbnail').status_code == 404
    assert client.get("/api/lora/'%20OR%201=1--/thumbnail").status_code == 404
    # An extra query parameter cannot choose a file instead of the catalogue row.
    assert client.get('/api/lora/missing/thumbnail', params={'path': str(source)}).status_code == 404
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_missing_database_is_not_created(tmp_path, monkeypatch):
    database = tmp_path / 'absent.db'
    monkeypatch.setattr(api, 'DB_PATH', database)
    assert TestClient(api.app).get('/api/lora/known/thumbnail').status_code == 503
    assert not database.exists()
