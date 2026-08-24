from click.testing import CliRunner

from ep_mcp import cli as cli_module
from ep_mcp.pack.loader import PackLoadError


def test_validate_error_output_is_ascii_safe(monkeypatch):
    def fail(_path):
        raise PackLoadError("invalid pack")

    monkeypatch.setattr(cli_module, "load_pack", fail)
    result = CliRunner().invoke(cli_module.cli, ["validate", "--pack", "missing-pack"])
    assert result.exit_code == 1
    assert "[ERROR] Pack load failed: invalid pack" in result.output
    assert "❌" not in result.output
