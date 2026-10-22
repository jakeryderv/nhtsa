import hashlib
import io
import threading
import zipfile
from datetime import UTC, datetime, timedelta
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nhtsa.download import download_complaints


def make_zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("FLAT_CMPL.txt", text)
    return buf.getvalue()


@pytest.fixture
def server():
    """Serve in-memory files like NHTSA's server, counting full downloads.

    Like the real server, it honors If-Modified-Since and ignores If-None-Match.
    A file's Last-Modified is derived from its content, so changing it bumps the date.
    """
    files = {"FLAT_CMPL.zip": make_zip("v1"), "CMPL.txt": b"fields"}
    downloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = files[self.path.lstrip("/")]
            digest = hashlib.md5(body).hexdigest()
            mtime = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
                seconds=int(digest[:6], 16)
            )
            last_modified = formatdate(mtime.timestamp(), usegmt=True)
            if self.headers.get("If-Modified-Since") == last_modified:
                self.send_response(304)
                self.end_headers()
                return
            downloads.append(self.path)
            self.send_response(200)
            self.send_header("ETag", f'"{digest}"')
            self.send_header("Last-Modified", last_modified)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", files, downloads
    httpd.shutdown()


def test_rerun_skips_unchanged_files(server, tmp_path):
    url, _, downloads = server
    download_complaints(tmp_path, base_url=url)
    assert sorted(downloads) == ["/CMPL.txt", "/FLAT_CMPL.zip"]
    assert (tmp_path / "FLAT_CMPL.txt").read_text() == "v1"
    assert (tmp_path / "manifest.json").exists()

    downloads.clear()
    download_complaints(tmp_path, base_url=url)
    assert downloads == []


def test_changed_file_is_redownloaded_and_reextracted(server, tmp_path):
    url, files, downloads = server
    download_complaints(tmp_path, base_url=url)
    files["FLAT_CMPL.zip"] = make_zip("v2")

    downloads.clear()
    download_complaints(tmp_path, base_url=url)
    assert downloads == ["/FLAT_CMPL.zip"]
    assert (tmp_path / "FLAT_CMPL.txt").read_text() == "v2"


def test_missing_local_files_are_restored(server, tmp_path):
    url, _, downloads = server
    download_complaints(tmp_path, base_url=url)
    (tmp_path / "CMPL.txt").unlink()
    (tmp_path / "FLAT_CMPL.txt").unlink()

    downloads.clear()
    download_complaints(tmp_path, base_url=url)
    assert downloads == ["/CMPL.txt"]
    assert (tmp_path / "FLAT_CMPL.txt").read_text() == "v1"


def test_force_redownloads(server, tmp_path):
    url, _, downloads = server
    download_complaints(tmp_path, base_url=url)

    downloads.clear()
    download_complaints(tmp_path, force=True, base_url=url)
    assert sorted(downloads) == ["/CMPL.txt", "/FLAT_CMPL.zip"]
