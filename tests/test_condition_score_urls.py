"""Tests for large condition-score URL lists."""

from app.engine.image_condition.services.image_filter import (
    prepare_condition_score_urls,
    sample_urls_evenly,
)


def test_sample_urls_evenly_spreads_across_feed() -> None:
    urls = [f"https://cdn.example/{i}.jpg" for i in range(100)]
    picked = sample_urls_evenly(urls, 10)
    assert len(picked) == 10
    assert picked[0] == urls[0]
    assert picked[-1] == urls[-1]
    assert len(set(picked)) == 10


def test_prepare_caps_at_max_input(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "CONDITION_SCORE_MAX_INPUT_URLS", 60)
    urls = [f"https://cdn.example/{i}.jpg" for i in range(90)]
    to_process, received, truncated = prepare_condition_score_urls(urls)
    assert received == 90
    assert len(to_process) == 60
    assert truncated == 30


def test_prepare_keeps_all_when_under_cap(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "CONDITION_SCORE_MAX_INPUT_URLS", 60)
    urls = [f"https://cdn.example/{i}.jpg" for i in range(30)]
    to_process, received, truncated = prepare_condition_score_urls(urls)
    assert received == 30
    assert len(to_process) == 30
    assert truncated == 0
