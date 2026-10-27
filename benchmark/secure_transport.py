"""Optional public-CA + leaf fingerprint pin, checked before HTTP headers."""
import re
import ssl

import requests


def pinned_session(fingerprint):
    fingerprint = fingerprint.replace(':', '').lower()
    if not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
        raise ValueError('Expected a SHA256 certificate fingerprint')
    context = ssl.create_default_context(cafile=requests.certs.where())
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True

    class PinnedAdapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs.update(ssl_context=context, assert_fingerprint=fingerprint)
            return super().init_poolmanager(*args, **kwargs)

        def proxy_manager_for(self, proxy, **kwargs):
            # urllib3 verifies the endpoint fingerprint during TLS connect,
            # before it sends the Authorization-bearing HTTP request.
            kwargs.update(ssl_context=context, assert_fingerprint=fingerprint)
            return super().proxy_manager_for(proxy, **kwargs)

    session = requests.Session()
    session.mount('https://', PinnedAdapter())
    return session
