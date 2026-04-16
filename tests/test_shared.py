from src.shared import Span, WordVersion, detect_version


class TestSpan:
    def test_len(self):
        assert len(Span(0, 10)) == 10

    def test_len_zero(self):
        assert len(Span(5, 5)) == 0

    def test_overlaps_true(self):
        assert Span(0, 10).overlaps(Span(5, 15))

    def test_overlaps_adjacent_is_false(self):
        # [0, 5) and [5, 10) share only a boundary — not overlapping
        assert not Span(0, 5).overlaps(Span(5, 10))

    def test_overlaps_contained(self):
        assert Span(0, 20).overlaps(Span(5, 10))

    def test_overlaps_symmetric(self):
        a, b = Span(0, 10), Span(8, 20)
        assert a.overlaps(b) == b.overlaps(a)

    def test_no_overlap(self):
        assert not Span(0, 5).overlaps(Span(10, 20))


class TestDetectVersion:
    def test_legacy_when_no_extended(self):
        names = ["word/document.xml", "word/comments.xml"]
        assert detect_version(names) == WordVersion.LEGACY

    def test_extended_when_only_extended(self):
        names = ["word/commentsExtended.xml"]
        assert detect_version(names) == WordVersion.EXTENDED

    def test_modern_when_both(self):
        names = ["word/commentsExtended.xml", "word/commentsIds.xml"]
        assert detect_version(names) == WordVersion.MODERN

    def test_empty_list_is_legacy(self):
        assert detect_version([]) == WordVersion.LEGACY
