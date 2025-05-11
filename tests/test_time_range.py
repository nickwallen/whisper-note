from datetime import datetime
from time_range import TimeRangeExtractor


class MockLangModel:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt):
        return self._response


def test_extract_good_response():
    response = '{"start": "2025-05-01", "end": "2025-05-03"}'
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("What did I do last week?")
    assert result.start == datetime(2025, 5, 1)
    assert result.end == datetime(2025, 5, 3, 23, 59, 59)


def test_extract_null_response():
    response = '{"start": null, "end": null}'
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("What is the weather?")
    assert result.start is None
    assert result.end is None


def test_extract_partial_response():
    response = '{"start": "2025-05-01", "end": null}'
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("What happened on May 1?")
    assert result.start == datetime(2025, 5, 1)
    assert result.end is None


def test_extract_bad_json():
    response = "not a json string"
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("Bad json response")
    assert result.start is None
    assert result.end is None


def test_extract_missing_fields():
    response = "{}"
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("Missing fields")
    assert result.start is None
    assert result.end is None


def test_extract_only_date_in_input():
    response = '{"start": "2025-05-01", "end": "2025-05-03"}'
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("Test only date in input")
    assert result.start == datetime(2025, 5, 1, 0, 0, 0)
    assert result.end == datetime(2025, 5, 3, 23, 59, 59)


def test_extract_invalid_date_format():
    response = '{"start": "2025/05/01", "end": "2025-05-03"}'
    extractor = TimeRangeExtractor(MockLangModel(response))
    result = extractor.extract("Invalid date format")
    assert result.start is None
    assert result.end == datetime(2025, 5, 3, 23, 59, 59)
