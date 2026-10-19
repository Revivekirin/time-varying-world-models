import hashlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('lock_report', Path(__file__).resolve().parents[1]/'scripts/lock_from_report.py')
lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock)
URL = 'https://download.pytorch.org/whl/jinja2-3.1.6-py3-none-any.whl'
SHA = 'a'*64

class LockReportTests(unittest.TestCase):
    def test_modern_hash(self):
        self.assertEqual(lock.artifact_hash({'url':URL,'archive_info':{'hashes':{'sha256':SHA}}}), (URL,SHA,False))
    def test_legacy_hash(self):
        self.assertEqual(lock.artifact_hash({'url':URL,'archive_info':{'hash':'sha256='+SHA}}), (URL,SHA,False))
    def test_fragment(self):
        self.assertEqual(lock.artifact_hash({'url':URL+'#sha256='+SHA}), (URL,SHA,False))
    def test_conflicting_or_malformed_hash(self):
        for value in ['b'*64,'broken']:
            with self.assertRaises(ValueError):
                lock.artifact_hash({'url':URL+'#sha256='+value,'archive_info':{'hashes':{'sha256':SHA}}}, True)
    def test_missing_is_explicit(self):
        with self.assertRaises(ValueError): lock.artifact_hash({'url':URL})
    def test_download_fingerprint(self):
        response = io.BytesIO(b'fixture bytes')
        response.geturl = lambda: URL
        with patch.object(lock,'urlopen',return_value=response) as fetch:
            self.assertEqual(lock.artifact_hash({'url':URL},True), (URL,hashlib.sha256(b'fixture bytes').hexdigest(),True))
            fetch.assert_called_once()
    def test_origin_rejected(self):
        with self.assertRaises(ValueError): lock.artifact_hash({'url':'https://example.com/test.whl'},True)
