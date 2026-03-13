import pytest
from click.testing import CliRunner
from unittest.mock import patch, AsyncMock


def test_cli_help():
    from hormuz.app.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "status" in result.output


def test_cli_init_db(tmp_path):
    from hormuz.app.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, ["init-db", "--db-path", str(tmp_path / "test.db")])
    assert result.exit_code == 0
    assert (tmp_path / "test.db").exists()


def test_cli_status_no_data(tmp_path):
    from hormuz.app.cli import cli
    from hormuz.infra.db import init_db
    db = tmp_path / "test.db"
    init_db(db)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db-path", str(db)])
    assert result.exit_code == 0
    assert "No data" in result.output or "no runs" in result.output.lower()
