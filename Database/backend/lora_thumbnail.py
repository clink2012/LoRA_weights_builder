"""Read only existing raster sidecars belonging to a catalogue LoRA."""
from pathlib import Path

MAX_THUMBNAIL_BYTES = 8 * 1024 * 1024
FORMATS = ((".jpeg", "image/jpeg"), (".jpg", "image/jpeg"),
           (".png", "image/png"), (".webp", "image/webp"))


class ThumbnailError(ValueError):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


def _raster_mime(content):
    # Downloaded sidecars can retain a .jpeg name while containing WebP/PNG.
    # Serve the actual raster type; never trust an extension as MIME authority.
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def read_thumbnail(file_path, root, *, max_bytes=MAX_THUMBNAIL_BYTES):
    """Return bounded image bytes and MIME; never accept a client file path.

    The caller supplies the authoritative catalogue path and configured root.
    Raster signatures constrain response types; decoding remains the browser's
    responsibility, so corrupt images still use the UI's missing-image fallback.
    """
    if not file_path:
        raise ThumbnailError(404, "No local thumbnail is available.")
    try:
        root = Path(root).resolve(strict=True)
        source = Path(file_path).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise ThumbnailError(404, "No local thumbnail is available.")
        for infix in ("", ".preview"):
            for extension, _extension_mime in FORMATS:
                candidate = source.with_name(source.stem + infix + extension)
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(root) or resolved.parent != source.parent:
                    continue
                if resolved.stat().st_size > max_bytes:
                    raise ThumbnailError(413, "The local thumbnail exceeds the image size limit.")
                # Limit the read itself too: a growing file must not bypass stat.
                with resolved.open("rb") as stream:
                    content = stream.read(max_bytes + 1)
                if len(content) > max_bytes:
                    raise ThumbnailError(413, "The local thumbnail exceeds the image size limit.")
                mime = _raster_mime(content)
                if mime is None:
                    raise ThumbnailError(415, "The local thumbnail is not a supported raster image.")
                return content, mime
    except (OSError, RuntimeError) as exc:
        raise ThumbnailError(404, "No local thumbnail is available.") from exc
    raise ThumbnailError(404, "No local thumbnail is available.")
