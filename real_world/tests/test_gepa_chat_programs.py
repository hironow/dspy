import pytest

import dspy
from real_world.dummy_lm import configure_dummy_adapter, make_dummy_lm_json
from real_world.gepa_chat_programs import (
    ProgramRequest,
    get_program,
    list_programs,
)


def _make_cycle_responder(responses):
    while True:
        yield from responses


@pytest.fixture(autouse=True)
def reset_dspy_settings():
    try:
        yield
    finally:
        dspy.settings.configure(lm=None, adapter=None)


def test_registry_contains_basic_program():
    slugs = {descriptor.slug for descriptor in list_programs()}
    assert "simple_gepa_basic" in slugs


def test_simple_gepa_basic_with_dummy_lm():
    program = get_program("simple_gepa_basic")

    lm = make_dummy_lm_json(
        _make_cycle_responder(
            [
                {"refined_question": "空の色は何色？"},
                {"answer": "青"},
            ]
        )
    )
    configure_dummy_adapter(lm=lm)

    request = ProgramRequest(prompt="空の色は何色ですか？", history=[], lm=lm)
    response = program.run(request)
    assert "青" in response.text


def test_langextract_program_fallback_outputs_extractions():
    program = get_program("simple_gepa_langextract")
    text = "Romeo gazed longingly at the stars while Juliet is the sun."
    lm = make_dummy_lm_json(
        _make_cycle_responder(
            [
                {
                    "prompt_description": "Extract characters and relationships.",
                    "examples_json": "[]",
                }
            ]
        )
    )
    configure_dummy_adapter(lm=lm)

    request = ProgramRequest(prompt=text, history=[], lm=lm)
    response = program.run(request)
    assert "Juliet" in response.text


def test_unsupported_program_raises():
    with pytest.raises(RuntimeError):
        get_program("simple_gepa_multimodal_caption")
