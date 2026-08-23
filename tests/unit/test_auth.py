from ep_mcp.auth import APIKeyAuth


def test_environment_key_is_registered_for_empty_config(monkeypatch):
    monkeypatch.setenv("EP_MCP_KEY_ALEX_HORMOZI_BRAIN", "secret")
    auth = APIKeyAuth()
    # build_app calls add_pack_keys for every pack, including [] keys; this
    # mirrors that registration path and protects the company deployment gate.
    auth.add_pack_keys("alex-hormozi-brain", [])
    assert auth.authenticate("Bearer secret", "alex-hormozi-brain")
    assert not auth.authenticate("Bearer wrong", "alex-hormozi-brain")
