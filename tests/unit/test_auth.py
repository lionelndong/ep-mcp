from ep_mcp.auth import APIKeyAuth
from ep_mcp.config import PackConfig, RateLimitConfig, ServerConfig
from ep_mcp.server import _TokenBucketRateLimiter, build_app
from types import SimpleNamespace

import httpx
import pytest


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


@pytest.mark.asyncio
async def test_network_build_app_rejects_missing_pack_key(monkeypatch):
    monkeypatch.delenv("EP_MCP_KEY_ALEX_HORMOZI_BRAIN", raising=False)

    class DummyMCP:
        session_manager = object()

        def streamable_http_app(self, **_kwargs):
            async def app(scope, receive, send):
                from starlette.responses import JSONResponse
                await JSONResponse({"ok": True})(scope, receive, send)
            return app

    pack = SimpleNamespace(name="Hormozi", type="person", version="1.0.0", files=[])
    instance = SimpleNamespace(pack=pack, mcp=DummyMCP())
    config = ServerConfig(
        host="0.0.0.0",
        packs=[PackConfig(slug="alex-hormozi-brain", path=".")],
        rate_limit=RateLimitConfig(enabled=True, requests_per_minute=60, burst=1),
    )
    app = build_app(config, {"alex-hormozi-brain": instance})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/search", params={"q": "offers", "pack": "alex-hormozi-brain"})
        mcp_response = await client.get("/packs/alex-hormozi-brain/mcp")
    assert response.status_code == 401
    assert mcp_response.status_code == 401
