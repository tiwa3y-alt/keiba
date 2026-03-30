"""Tests for scraper utility functions."""

from keiba.scraper.netkeiba import _parse_time_to_seconds, _safe_float, _safe_int


class TestParseTime:
    def test_standard_format(self):
        assert _parse_time_to_seconds("1:34.5") == 94.5

    def test_dot_separator(self):
        # "1.34.5" should also be parsed (first dot → colon)
        assert _parse_time_to_seconds("1.34.5") == 94.5

    def test_empty(self):
        assert _parse_time_to_seconds("") is None
        assert _parse_time_to_seconds("-") is None
        assert _parse_time_to_seconds(None) is None

    def test_seconds_only(self):
        assert _parse_time_to_seconds("34.5") == 34.5


class TestSafeInt:
    def test_normal(self):
        assert _safe_int("123") == 123

    def test_with_comma(self):
        assert _safe_int("1,234") == 1234

    def test_negative(self):
        assert _safe_int("-5") == -5

    def test_invalid(self):
        assert _safe_int("abc") is None
        assert _safe_int("") is None


class TestSafeFloat:
    def test_normal(self):
        assert _safe_float("3.5") == 3.5

    def test_with_yen(self):
        assert _safe_float("¥100.5") == 100.5

    def test_invalid(self):
        assert _safe_float("abc") is None
