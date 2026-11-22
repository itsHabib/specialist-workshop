import hashlib
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from urllib.request import Request

from decision_workshop import base


class PinnedBaseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.out = self.root / "base"
        self.bodies = {name: f"test bytes for {name}".encode() for name in base.FILE_HASHES}
        pins = {name: hashlib.sha256(body).hexdigest() for name, body in self.bodies.items()}
        pin_patch = patch.object(base, "FILE_HASHES", pins)
        pin_patch.start()
        self.addCleanup(pin_patch.stop)
        opener_patch = patch.object(base, "build_opener")
        self.opener = opener_patch.start()
        self.addCleanup(opener_patch.stop)
        self.open = self.opener.return_value.open
        self.open.side_effect = self.serve_fixture

    def serve_fixture(self, url, timeout):
        prefix = f"https://huggingface.co/{base.MODEL_ID}/resolve/{base.REVISION}/"
        self.assertTrue(url.startswith(prefix))
        return io.BytesIO(self.bodies[url.removeprefix(prefix)])

    def test_prepare_verify_and_repeat_offline(self):
        self.assertEqual(base.prepare_base(self.out), self.out)
        self.assertEqual(self.open.call_count, 3)
        self.opener.reset_mock()
        self.assertEqual(base.prepare_base(self.out), self.out)
        self.assertEqual(base.verify_base(self.out), self.out)
        self.opener.assert_not_called()
        (self.out / "vocab.txt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, r"vocab.txt: SHA-256 mismatch"):
            base.prepare_base(self.out)
        self.opener.assert_not_called()
        self.assertEqual((self.out / "vocab.txt").read_bytes(), b"tampered")

    def test_unknown_missing_and_linked_inputs(self):
        base.prepare_base(self.out)
        (self.out / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "extra.*tokenizer.json"):
            base.verify_base(self.out)
        (self.out / "tokenizer.json").unlink()
        (self.out / "config.json").rename(self.root / "original")
        with self.assertRaisesRegex(ValueError, "missing.*config.json"):
            base.verify_base(self.out)
        (self.out / "config.json").symlink_to(self.root / "original")
        with self.assertRaisesRegex(ValueError, "regular checkpoint file"):
            base.verify_base(self.out)

    def test_corrupt_download_and_interruption_leave_no_partial_output(self):
        self.open.side_effect = lambda *a, **kw: io.BytesIO(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            base.prepare_base(self.out)
        self.assertEqual(list(self.root.iterdir()), [])
        self.open.side_effect = [io.BytesIO(self.bodies["config.json"]), OSError("disconnected")]
        with self.assertRaisesRegex(OSError, "disconnected"):
            base.prepare_base(self.out)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_independent_preparers_can_publish_the_same_checkpoint(self):
        barrier = Barrier(2, timeout=5)

        def serve_concurrently(url, timeout):
            if url.endswith("/config.json"):
                barrier.wait()
            return self.serve_fixture(url, timeout)

        self.open.side_effect = serve_concurrently
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(base.prepare_base, [self.out, self.out]))
        self.assertEqual(results, [self.out, self.out])
        self.assertEqual(list(self.root.iterdir()), [self.out])
        base.verify_base(self.out)

    def test_redirects_stay_on_hugging_face_https(self):
        handler = base._HubRedirects()
        request = Request("https://huggingface.co/repo/checkpoint")
        for url in ("http://huggingface.co/path", "https://huggingface.co.example.org/x",
                    "https://example.org/x", "https://example.org@huggingface.co/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, "", {}, url)
        target = "https://us.aws.cdn.hf.co/checkpoint"
        redirected = handler.redirect_request(request, None, 302, "", {}, target)
        self.assertEqual(redirected.full_url, target)
