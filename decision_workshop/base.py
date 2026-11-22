"""Acquire and verify the exact Google BERT-Tiny checkpoint for decision training.

Google Research release and license: https://github.com/google-research/bert
Google model repository: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
The pinned repository revision includes Hugging Face's safetensors conversion.
"""

import errno
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener


MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
FILE_HASHES = {
    "config.json": "508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "model.safetensors": "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414",
}


class _HubRedirects(HTTPRedirectHandler):
    """Allow the Hub to deliver pinned files through its own HTTPS CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        host = target.hostname or ""
        if (target.scheme != "https" or target.port not in (None, 443)
                or target.username is not None
                or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))):
            raise ValueError("checkpoint download redirected outside Hugging Face HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_base(path: Path) -> Path:
    """Verify locally, including the file list consumed by model/tokenizer loaders."""
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"expected a directory containing the pinned base at {path}")
    found = {entry.name for entry in path.iterdir()}
    missing = FILE_HASHES.keys() - found
    extra = found - FILE_HASHES.keys()
    if missing or extra:
        raise ValueError(f"{path}: wrong base files (missing {sorted(missing)}, extra {sorted(extra)})")
    for name, expected in FILE_HASHES.items():
        file = path / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"{file}: expected a regular checkpoint file")
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"{file}: SHA-256 mismatch for {MODEL_ID} at {REVISION}")
    return path


def prepare_base(out: Path) -> Path:
    """Explicitly download the public base or verify an existing copy offline."""
    out = Path(out).expanduser()
    if out.exists() or out.is_symlink():
        return verify_base(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(_HubRedirects())
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as staging:
        base = Path(staging) / "base"
        base.mkdir()
        for name in FILE_HASHES:
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
            with opener.open(url, timeout=60) as source, (base / name).open("xb") as dest:
                shutil.copyfileobj(source, dest)
        verify_base(base)
        try:
            base.rename(out)
        except OSError as exc:
            # A concurrent preparer may already have published the complete directory.
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                raise
            return verify_base(out)
    return out
