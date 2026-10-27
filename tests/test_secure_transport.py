"""Check TLS policy without sockets, credentials, or external requests."""
import ssl
import unittest

from benchmark.secure_transport import pinned_session


class PinnedTransportTests(unittest.TestCase):
    def test_pin_is_enforced_for_direct_and_proxy_connections(self):
        pin = 'ab' * 32
        with pinned_session(pin) as session:
            self.assertTrue(session.verify)
            adapter = session.get_adapter('https://chatgpt.com')
            for manager in (adapter.poolmanager, adapter.proxy_manager_for('http://127.0.0.1:7851')):
                self.assertEqual(manager.connection_pool_kw['assert_fingerprint'], pin)
                self.assertEqual(manager.connection_pool_kw['ssl_context'].verify_mode, ssl.CERT_REQUIRED)

    def test_invalid_pin_fails_before_any_network_io(self):
        for pin in ('', 'none', 'ab' * 31, 'zz' * 32):
            with self.assertRaises(ValueError):
                pinned_session(pin)


if __name__ == '__main__':
    unittest.main()
