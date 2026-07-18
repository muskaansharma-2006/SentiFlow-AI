from preprocessing.text_preprocessing import preprocess_text


def test_slang_and_noise_are_normalized():
    assert preprocess_text("OMG this app is fire!!! https://example.com") == "oh my god this app is excellent!"


def test_empty_input_is_safe():
    assert preprocess_text(None) == ""
