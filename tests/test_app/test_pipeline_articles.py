"""Tests for pipeline article storage and dedup."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from hormuz.infra.db import init_db, insert_articles, get_article_ids


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "test.db"
    init_db(p)
    return p


def test_pipeline_skips_already_processed_articles(db_path):
    """Articles already in DB should be filtered out before LLM extraction."""
    # Pre-insert one article
    insert_articles(db_path, [{"id": "old-1", "title": "Old news", "source": "Reuters"}])

    all_articles = [
        {"id": "old-1", "title": "Old news", "source": "Reuters", "summary": "x"},
        {"id": "new-1", "title": "New news", "source": "gCaptain", "summary": "y"},
        {"id": "new-2", "title": "Fresh news", "source": "Al Jazeera", "summary": "z"},
    ]

    existing = get_article_ids(db_path, {a["id"] for a in all_articles if a.get("id")})
    new_articles = [a for a in all_articles if a.get("id") not in existing]

    assert len(new_articles) == 2
    assert all(a["id"] != "old-1" for a in new_articles)
