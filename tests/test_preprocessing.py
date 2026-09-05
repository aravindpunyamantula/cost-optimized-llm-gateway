from app.core.preprocessing import preprocess_prompt


def test_strips_leading_and_trailing_whitespace():
    prompt = "   Hello world   "

    result = preprocess_prompt(prompt)

    assert result == "Hello world"


def test_reduces_multiple_newlines():
    prompt = "Hello\n\n\n\nWorld"

    result = preprocess_prompt(prompt)

    assert result == "Hello\n\nWorld"


def test_normalizes_carriage_returns():
    prompt = "Hello\r\nWorld\rTest"

    result = preprocess_prompt(prompt)

    assert result == "Hello\nWorld\nTest"


def test_removes_unprintable_characters():
    prompt = "Hello\x00World\x07"

    result = preprocess_prompt(prompt)

    assert result == "HelloWorld"


def test_preserves_normal_internal_spaces():
    prompt = "Hello   world"

    result = preprocess_prompt(prompt)

    assert result == "Hello   world"