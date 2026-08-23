from ep_mcp.auth import APIKeyAuth
from ep_mcp.server import _TokenBucketRateLimiter


def test_environment_key_is_registered_for_empty_config(monkeypatch):
    monkeypatch.setenv("EP_MCP_KEY_ALEX_HORMOZI_BRAIN", "secret")
    auth = APIKeyAuth()
    # build_app calls add_pack_keys for every pack, including [] keys; this
    # mirrors that registration path and protects the company deployment gate.
    auth.add_pack_keys("alex-hormozi-brain", [])
    assert auth.authenticate("Bearer secret", "alex-hormozi-brain")
    assert not auth.authenticate("Bearer wrong", "alex-hormozi-brain")


def test_network_auth_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("EP_MCP_KEY_ALEX_HORMOZI_BRAIN", raising=False)
    auth = APIKeyAuth(allow_open=False)
    auth.add_pack_keys("alex-hormozi-brain", [])
    assert not auth.authenticate("Bearer anything", "alex-hormozi-brain")


def test_rate_limiter_enforces_burst_and_retry_after():
    limiter = _TokenBucketRateLimiter(requests_per_minute=60, burst=2)
    assert limiter.allow("pack:client")[0]
    assert limiter.allow("pack:client")[0]
    allowed, retry_after = limiter.allow("pack:client")
    assert not allowed
    assert retry_after >= 1
