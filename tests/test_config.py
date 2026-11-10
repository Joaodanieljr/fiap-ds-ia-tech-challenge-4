import pytest

from voice_of_customer.config import load_settings


def test_load_settings_parses_optional_retrieval_max_distance(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "0.42")

    settings = load_settings()

    assert settings.retrieval_max_distance == 0.42


@pytest.mark.parametrize("value", ["-0.1", "2.1"])
def test_load_settings_rejects_out_of_range_retrieval_max_distance(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", value)

    with pytest.raises(ValueError, match="entre 0 e 2"):
        load_settings()


def test_load_settings_leaves_retrieval_max_distance_disabled_when_empty(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_MAX_DISTANCE", "")

    settings = load_settings()

    assert settings.retrieval_max_distance is None
