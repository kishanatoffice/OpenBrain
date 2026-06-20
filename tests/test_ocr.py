"""OCR connector — security envelope + behavior.

The conversion itself (markitdown) is monkeypatched so these run without the
heavy optional dependency. What matters here is the envelope around it: path
confinement, URL refusal, size/type caps, redaction on ingest, provenance, and
the disabled-by-default switch.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
import zipfile
from pathlib import Path

from myagent import ocr
from myagent.connectors import REGISTRY, default_enabled_keys
from myagent.mcp import call_tool

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


class OcrEnvelopeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Resolve the root: on macOS /var/folders symlinks to /private/var, and
        # the connector compares against resolved paths.
        self.root = Path(self.tmp.name).resolve()
        self.deps = make_deps(self.tmp.name)
        self.deps.ocr_ingest_dirs = (self.root,)
        self.deps.ocr_max_bytes = 1024
        # Make conversion deterministic and dependency-free.
        self._orig = ocr._convert_bytes
        self._orig_dims = ocr._image_pixels
        ocr._convert_bytes = lambda data, ext: "converted-body-text"

    def tearDown(self):
        ocr._convert_bytes = self._orig
        ocr._image_pixels = self._orig_dims
        self.tmp.cleanup()

    def _write(self, name, content="hello", outside=False):
        base = Path(tempfile.gettempdir()) if outside else self.root
        p = base / name
        p.write_text(content, encoding="utf-8")
        return p

    def _digest(self, **args):
        return run(ocr.digest_handler(self.deps, "digest", args))

    # ---- the switch ----------------------------------------------------------

    def test_disabled_by_default(self):
        self.assertNotIn("ocr", default_enabled_keys())
        c = REGISTRY["ocr"]
        self.assertTrue(c.toggleable)
        self.assertFalse(c.default_enabled)

    def test_call_tool_rejects_when_connector_off(self):
        # enabled_connectors=None -> registry defaults -> ocr is OFF.
        out = run(call_tool(self.deps, "digest", {"path": "x"}))
        self.assertIn("switched off", out)

    # ---- path confinement ----------------------------------------------------

    def test_file_outside_allowlist_refused(self):
        p = self._write("outside.txt", outside=True)
        try:
            out = self._digest(path=str(p))
            self.assertIn("outside the allowlisted", out)
        finally:
            p.unlink()

    def test_url_refused(self):
        out = self._digest(path="https://evil.example/secret")
        self.assertIn("URL ingestion is disabled", out)

    def test_missing_file_refused(self):
        out = self._digest(path=str(self.root / "nope.txt"))
        self.assertIn("file not found", out)

    def test_symlink_escape_refused(self):
        # A symlink *inside* the root pointing *outside* must not bypass the
        # check — resolve() collapses it before the containment test.
        secret = self._write("secret.txt", content="x", outside=True)
        link = self.root / "link.txt"
        try:
            link.symlink_to(secret)
            out = self._digest(path=str(link))
            self.assertIn("outside the allowlisted", out)
        finally:
            link.unlink(missing_ok=True)
            secret.unlink(missing_ok=True)

    def test_no_ingest_dir_configured_refused(self):
        self.deps.ocr_ingest_dirs = ()
        p = self._write("a.txt")
        out = self._digest(path=str(p))
        self.assertIn("no ingest directory", out)

    # ---- caps ----------------------------------------------------------------

    def test_oversize_refused(self):
        p = self._write("big.txt", content="x" * 2048)  # > 1024 cap
        out = self._digest(path=str(p))
        self.assertIn("over the", out)

    def test_unsupported_extension_refused(self):
        p = self._write("payload.exe")
        out = self._digest(path=str(p))
        self.assertIn("unsupported file type", out)

    # ---- behavior ------------------------------------------------------------

    def test_happy_path_returns_markdown_with_provenance(self):
        p = self._write("doc.txt")
        out = self._digest(path=str(p))
        self.assertIn("# Digest of doc.txt", out)
        self.assertIn("low-confidence", out)
        self.assertIn("converted-body-text", out)
        self.assertIn("UNTRUSTED INGESTED DOCUMENT", out)  # always fenced

    def test_redaction_runs_on_ingest(self):
        ocr._convert_bytes = lambda data, ext: "key sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        p = self._write("leak.txt")
        out = self._digest(path=str(p))
        self.assertNotIn("sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", out)
        self.assertIn("redacted", out)

    def test_visual_image_warns_on_sparse_text(self):
        ocr._convert_bytes = lambda data, ext: "x"  # below the text floor
        ocr._image_pixels = lambda data: None  # not a real decodable image
        p = self._write("diagram.png")
        out = self._digest(path=str(p))
        self.assertIn("vision model", out)

    def test_save_as_memory_persists(self):
        p = self._write("notes.txt")
        out = self._digest(path=str(p), save_as_memory=True)
        self.assertIn("Saved to memory", out)
        self.assertEqual(self.deps.store.count(), 1)

    def test_saved_memory_is_tagged_untrusted(self):
        p = self._write("notes.txt")
        self._digest(path=str(p), save_as_memory=True)
        rows = self.deps.store.browse(limit=10)["rows"]
        self.assertEqual(len(rows), 1)
        self.assertIn("untrusted", rows[0]["tags"])
        self.assertTrue(any(t.startswith("source:") for t in rows[0]["tags"]))

    def test_filename_injection_is_sanitized(self):
        # A crafted filename must not inject markdown/HTML or break the tag.
        ocr._convert_bytes = lambda data, ext: "body"
        p = self._write("a b#x`y,z.txt")  # nasty but filesystem-legal chars
        out = self._digest(path=str(p), save_as_memory=True)
        self.assertNotIn("#x", out)        # '#' was stripped from the name
        rows = self.deps.store.browse(limit=10)["rows"]
        src = [t for t in rows[0]["tags"] if t.startswith("source:")][0]
        self.assertNotIn("#", src)
        self.assertNotIn(",", src)         # tag can't be split

    def test_image_reports_estimated_savings(self):
        from myagent import tokens
        ocr._convert_bytes = lambda data, ext: "some OCR'd text from the screenshot"
        ocr._image_pixels = lambda data: (1600, 1200)  # large image
        before = tokens.tokens_saved()
        p = self._write("screenshot.png")
        out = self._digest(path=str(p))
        self.assertIn("vision tokens", out)
        self.assertIn("saving", out)
        self.assertGreater(tokens.tokens_saved(), before)  # counter moved

    def test_oversize_image_pixels_refused(self):
        ocr._image_pixels = lambda data: (50_000, 50_000)  # 2.5 Gpx bomb
        p = self._write("bomb.png")
        out = self._digest(path=str(p))
        self.assertIn("MP decode limit", out)

    def test_document_makes_no_savings_claim(self):
        # A .txt document has no honest vision-token baseline — no claim made.
        ocr._convert_bytes = lambda data, ext: "plain document body"
        p = self._write("report.txt")
        out = self._digest(path=str(p))
        self.assertNotIn("vision tokens", out)
        self.assertIn("tokens of text", out)  # cost reported, not savings

    def test_nul_byte_path_refused_cleanly(self):
        out = self._digest(path="foo\x00bar.txt")
        self.assertIn("invalid path", out)  # clean refusal, no stack trace

    def test_output_truncated_at_cap(self):
        ocr._convert_bytes = lambda data, ext: "A" * (ocr._MAX_OUTPUT_CHARS + 500)
        p = self._write("huge.txt", content="x")  # source small; output huge
        out = self._digest(path=str(p))
        self.assertIn("truncated", out)
        # The huge body itself is bounded (plus small header/notes/footer/fence).
        self.assertLess(len(out), ocr._MAX_OUTPUT_CHARS + 2000)


class OcrZipBombCase(unittest.TestCase):
    """The expanded-size / ratio / entry-count guard for ZIP-container files
    (Office docs, EPubs, or anything secretly a zip)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.deps = make_deps(self.tmp.name)
        self.deps.ocr_ingest_dirs = (self.root,)
        self.deps.ocr_max_bytes = 50 * 1024 * 1024  # let the source through;
        self.deps.ocr_max_expanded_bytes = 1 * 1024 * 1024  # the EXPANDED cap bites
        # If conversion is ever reached it would mean the guard failed; make it loud.
        self._orig = ocr._convert_bytes
        ocr._convert_bytes = lambda data, ext: "SHOULD-NOT-CONVERT"

    def tearDown(self):
        ocr._convert_bytes = self._orig
        self.tmp.cleanup()

    def _zip(self, name, entries):
        p = self.root / name
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            for ename, data in entries:
                zf.writestr(ename, data)
        return p

    def _digest(self, p):
        return run(ocr.digest_handler(self.deps, "digest", {"path": str(p)}))

    def test_expanded_size_over_cap_refused(self):
        # 2 MB of zeros compresses to ~nothing but expands past the 1 MB cap.
        p = self._zip("bomb.docx", [("a.bin", b"\0" * (2 * 1024 * 1024))])
        out = self._digest(p)
        self.assertIn("expands to", out)
        self.assertNotIn("SHOULD-NOT-CONVERT", out)  # never parsed

    def test_high_ratio_refused(self):
        # Under the expanded cap but absurd compression ratio.
        self.deps.ocr_max_expanded_bytes = 100 * 1024 * 1024
        p = self._zip("ratio.docx", [("a.bin", b"\0" * (900 * 1024))])
        out = self._digest(p)
        self.assertIn("ratio", out)

    def test_too_many_entries_refused(self):
        self.deps.ocr_max_expanded_bytes = 100 * 1024 * 1024
        entries = [(f"f{i}.txt", b"") for i in range(ocr._ZIP_MAX_ENTRIES + 1)]
        p = self._zip("many.docx", entries)
        out = self._digest(p)
        self.assertIn("entries", out)

    def test_benign_zip_passes_guard(self):
        ocr._convert_bytes = lambda data, ext: "real content"
        p = self._zip("ok.docx", [("word/document.xml", b"<xml>hi</xml>")])
        out = self._digest(p)
        self.assertIn("real content", out)


if __name__ == "__main__":
    unittest.main()
