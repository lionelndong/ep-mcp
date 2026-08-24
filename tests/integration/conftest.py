"""Pytest options shared by integration tests."""

from __future__ import annotations


def pytest_addoption(parser):
    """Register the optional real-pack path used by integration tests."""

    parser.addoption(
        "--pack",
        action="store",
        default=None,
        help="Path to an ExpertPack for external-embedding integration tests",
    )
