"""Tests for articles storage and provenance tracking."""

import pytest
from datetime import datetime
from pathlib import Path

from hormuz.infra.db import (
    init_db,
    insert_articles,
    get_article_ids,
    insert_article_observations,
    get_article_observations,
)


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    init_db(p)
    return p


def _make_article(id="art-1", title="Test", source="Reuters", url="http://example.com", summary="body"):
    return {"id": id, "title": title, "source": source, "url": url, "summary": summary, "published_date": "2026-03-14"}


def test_insert_and_retrieve_articles(db_path):
    """Articles are stored and can be queried by ID set."""
    articles = [_make_article("a1"), _make_article("a2"), _make_article("a3")]
    insert_articles(db_path, articles)
    existing = get_article_ids(db_path, {"a1", "a2", "a4"})
    assert existing == {"a1", "a2"}


def test_insert_articles_dedup(db_path):
    """Inserting same article ID twice doesn't duplicate."""
    insert_articles(db_path, [_make_article("a1")])
    insert_articles(db_path, [_make_article("a1", title="Updated")])
    existing = get_article_ids(db_path, {"a1"})
    assert existing == {"a1"}


def test_insert_article_observations(db_path):
    """Article→observation provenance is stored and retrievable."""
    insert_articles(db_path, [_make_article("a1")])
    mappings = [
        {"article_id": "a1", "obs_id": "O01", "confidence": "high"},
        {"article_id": "a1", "obs_id": "O03", "confidence": "medium"},
    ]
    insert_article_observations(db_path, mappings, batch_run="run-001")
    result = get_article_observations(db_path, obs_id="O01")
    assert len(result) == 1
    assert result[0]["article_id"] == "a1"
    assert result[0]["confidence"] == "high"
    assert result[0]["batch_run"] == "run-001"


def test_get_article_observations_by_batch(db_path):
    """Can filter provenance by batch_run."""
    insert_articles(db_path, [_make_article("a1"), _make_article("a2")])
    insert_article_observations(db_path, [
        {"article_id": "a1", "obs_id": "O01", "confidence": "high"},
    ], batch_run="run-001")
    insert_article_observations(db_path, [
        {"article_id": "a2", "obs_id": "O01", "confidence": "medium"},
    ], batch_run="run-002")
    r1 = get_article_observations(db_path, batch_run="run-001")
    assert len(r1) == 1
    assert r1[0]["article_id"] == "a1"
