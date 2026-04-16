from src.comments.extract import Comment
from src.stats.compute import comment_metrics, comment_metrics_from_df, latest_date


def _comment(id: str, date: str, resolved: bool = False, replies=None) -> Comment:
    c = Comment(id=id, author="Alice", date=date, text="text", resolved=resolved)
    c.replies = replies or []
    return c


class TestCommentMetrics:
    def test_empty(self):
        m = comment_metrics([])
        assert m.total == 0
        assert m.resolved == 0

    def test_single_unresolved(self):
        m = comment_metrics([_comment("1", "2024-01-01")])
        assert m.total == 1
        assert m.top_level == 1
        assert m.replies == 0
        assert m.resolved == 0

    def test_single_resolved(self):
        m = comment_metrics([_comment("1", "2024-01-01", resolved=True)])
        assert m.resolved == 1

    def test_with_replies(self):
        reply = _comment("2", "2024-01-02")
        parent = _comment("1", "2024-01-01", replies=[reply])
        m = comment_metrics([parent])
        assert m.total == 2
        assert m.top_level == 1
        assert m.replies == 1

    def test_resolved_reply_counted(self):
        reply = _comment("2", "2024-01-02", resolved=True)
        parent = _comment("1", "2024-01-01", replies=[reply])
        m = comment_metrics([parent])
        assert m.resolved == 1


class TestCommentMetricsFromDf:
    def test_empty_df(self):
        from pandas import DataFrame

        m = comment_metrics_from_df(DataFrame())
        assert m.total == 0
        assert m.top_level == 0
        assert m.replies == 0
        assert m.resolved == 0

    def test_non_empty_df(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "kind": ["comment", "reply", "comment"],
                "resolved": [False, True, False],
            }
        )
        m = comment_metrics_from_df(df)
        assert m.total == 3
        assert m.top_level == 2
        assert m.replies == 1
        assert m.resolved == 1


class TestLatestDate:
    def test_returns_none_for_empty(self):
        assert latest_date([], []) is None

    def test_returns_latest(self):
        c1 = _comment("1", "2024-01-01")
        c2 = _comment("2", "2024-06-15")
        result = latest_date([c1, c2], [])
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    def test_skips_unparseable_date(self):
        good = _comment("1", "2024-03-01")
        bad = _comment("2", "not-a-date")
        # Should not raise; should return the good date
        result = latest_date([good, bad], [])
        assert result is not None
        assert result.month == 3

    def test_all_bad_dates_returns_none(self):
        bad = _comment("1", "garbage")
        assert latest_date([bad], []) is None
