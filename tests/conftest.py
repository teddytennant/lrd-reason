"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from lrd_reason.config import RunSpec, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO_ROOT / "configs" / "smoke.yaml"
SMOKE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "cot.smoke.jsonl"


@pytest.fixture
def smoke_spec() -> RunSpec:
    return load_config(SMOKE_CONFIG)


@pytest.fixture
def smoke_fixture_path() -> Path:
    return SMOKE_FIXTURE
