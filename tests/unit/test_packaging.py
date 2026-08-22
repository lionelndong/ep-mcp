"""Tests that runtime imports are represented in the install metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_numpy_is_declared_as_a_runtime_dependency() -> None:
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]

    assert any(dependency.lower().startswith("numpy") for dependency in dependencies)
