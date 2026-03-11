"""Shared test fixtures for hormuz-ds."""
import sqlite3
from pathlib import Path
import pytest
import yaml

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"

@pytest.fixture
def constants() -> dict:
    path = Path(__file__).parent.parent / "configs" / "constants.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def parameters() -> dict:
    path = Path(__file__).parent.parent / "configs" / "parameters.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
