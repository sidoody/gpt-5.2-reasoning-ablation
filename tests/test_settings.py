from gpt_5_2_reasoning_ablation.settings import StudySettings


def test_variant_ids_are_stable():
    settings = StudySettings()
    variants = settings.variants(["none", "high"])
    assert [variant.variant_id for variant in variants] == [
        "gpt-5.2__reasoning-none",
        "gpt-5.2__reasoning-high",
    ]


def test_xhigh_is_rejected():
    settings = StudySettings()
    try:
        settings.variants(["none", "xhigh"])
    except ValueError as exc:
        assert "Unsupported reasoning level: xhigh" in str(exc)
    else:
        raise AssertionError("Expected StudySettings.variants() to reject xhigh")


def test_temperature_out_of_range_raises_validation_error():
    settings = StudySettings(temperature=2.5)

    try:
        settings.validate()
    except ValueError as exc:
        assert "temperature must be between 0 and 2 inclusive" in str(exc)
    else:
        raise AssertionError("Expected StudySettings.validate() to reject out-of-range temperature")
