"""Tests for per-room deduplication before vision scoring."""

from app.engine.image_condition.services.image_filter import FilteredImage, deduplicate_filtered_by_room_type


def _img(url: str, room: str, conf: float, rankings: tuple[tuple[str, float], ...]) -> FilteredImage:
    return FilteredImage(
        image_url=url,
        room_type=room,
        confidence=conf,
        image_bytes=b"x" * 512,
        room_rankings=rankings,
    )


def test_dedupe_fills_distinct_rooms_from_ranked_labels() -> None:
    images = [
        _img(
            "https://cdn/a.jpg",
            "exterior front of house",
            0.45,
            (
                ("exterior front of house", 0.45),
                ("kitchen interior", 0.18),
            ),
        ),
        _img(
            "https://cdn/b.jpg",
            "exterior front of house",
            0.42,
            (
                ("exterior front of house", 0.42),
                ("bathroom interior", 0.16),
            ),
        ),
        _img(
            "https://cdn/c.jpg",
            "exterior front of house",
            0.40,
            (
                ("exterior front of house", 0.40),
                ("bedroom interior", 0.15),
            ),
        ),
        _img(
            "https://cdn/d.jpg",
            "kitchen interior",
            0.35,
            (("kitchen interior", 0.35),),
        ),
    ]
    unique, skipped = deduplicate_filtered_by_room_type(images)
    rooms = {row.room_type for row in unique}
    assert rooms == {
        "exterior front of house",
        "kitchen interior",
        "bathroom interior",
        "bedroom interior",
    }
    assert len(unique) == 4
    assert skipped == 0


def test_dedupe_fallback_when_clip_unknown_room_types() -> None:
    images = [
        _img(f"https://cdn/{i}.jpg", "unknown", 0.5, ())
        for i in range(10)
    ]
    unique, skipped = deduplicate_filtered_by_room_type(images)
    assert len(unique) == 6
    assert skipped == 4


def test_dedupe_keeps_one_per_duplicate_room_when_no_alternate_labels() -> None:
    images = [
        _img("https://cdn/1.jpg", "bedroom interior", 0.5, (("bedroom interior", 0.5),)),
        _img("https://cdn/2.jpg", "bedroom interior", 0.3, (("bedroom interior", 0.3),)),
    ]
    unique, skipped = deduplicate_filtered_by_room_type(images)
    assert len(unique) == 1
    assert unique[0].room_type == "bedroom interior"
    assert skipped == 1
