"""Portable configuration must not read credentials or weaken isolation."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.codex_isolated_run import resolve_auth_file, sandbox_command, validate_codex_binary


class PortabilityTests(unittest.TestCase):
    def test_auth_file_selection_without_reading_credentials(self):
        with patch.dict(os.environ, {'CODEX_HOME': '/example/account'}):
            self.assertEqual(resolve_auth_file(), Path('/example/account/auth.json'))
            self.assertEqual(resolve_auth_file('/explicit/auth.json'), Path('/explicit/auth.json'))
        with patch.dict(os.environ, {}, clear=True), patch('pathlib.Path.home', return_value=Path('/example/user')):
            self.assertEqual(resolve_auth_file(), Path('/example/user/.codex/auth.json'))

    def test_native_binary_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / 'codex'
            launcher.write_text('#!/usr/bin/env node\n')
            launcher.chmod(0o700)
            with self.assertRaises(ValueError):
                validate_codex_binary(launcher)

    def test_sandbox_keeps_namespace_boundary_and_explicit_host_probes(self):
        with patch('benchmark.codex_isolated_run.validate_codex_binary', return_value=Path('/custom/codex')), patch('shutil.which', return_value='/usr/bin/bwrap'):
            command = sandbox_command('/tmp/broker', '/tmp/entry', 'example-model', '/custom/codex', ['/private/project', '/private/home', '/private/data'])
        self.assertIn('--unshare-all', command)
        self.assertIn('--clearenv', command)
        self.assertIn('/custom/codex', command)
        self.assertIn('/opt/codex', command)
        self.assertIn('ISOLATION_HIDDEN_PATHS', command)
        self.assertNotIn('/private/data', command)  # Probe metadata, never a mount.
        self.assertFalse(any('auth.json' in item for item in command))
