import json
from typing import Any

import pytest

import dspy
from real_world import data_tools as DT
from real_world import factory as F


def test_io_and_cleaning_roundtrip(tmp_path):
    """
    Given a small CSV/JSONL dataset
    When we load, clean, rename, select, dedupe and save
    Then the rows persist correctly and shapes match expectations
    """
    # Given
    jsonl_path = tmp_path / "sample.jsonl"
    rows_in = [
        {"question": "What?", "answer": "Yes"},
        {"question": "What?", "answer": "Yes"},
        {"question": "  Why  ", "answer": "  No  "},
    ]
    DT.save_jsonl(jsonl_path, rows_in)

    # When
    rows = DT.load_jsonl(jsonl_path)
    rows = DT.normalize_text(rows, fields=["question", "answer"])  # trim and collapse whitespace
    rows = DT.rename_keys(rows, {"question": "q", "answer": "a"})
    rows = DT.select_keys(rows, ["q", "a"])  # enforce a small schema
    rows = DT.dedupe(rows, by_keys=["q", "a"])  # remove exact dupes
    rows = DT.drop_missing(rows, required_keys=["q", "a"])  # keep complete rows

    # Then
    assert len(rows) == 2
    assert rows[0]["q"] == "What?" and rows[0]["a"] == "Yes"
    assert rows[1]["q"] == "Why" and rows[1]["a"] == "No"

    # Roundtrip JSONL
    jsonl_path = tmp_path / "out.jsonl"
    DT.save_jsonl(jsonl_path, rows)
    loaded = DT.load_jsonl(jsonl_path)
    assert loaded == rows


@pytest.mark.parametrize(
    "name, make_rows, input_keys, factory_builder",
    [
        (
            "qa",
            lambda: [
                {"question": "空の色は何色ですか？", "answer": "青"},
                {"question": "バナナの色は何色ですか？", "answer": "黄色"},
            ],
            ["question"],
            lambda: F.basic_qa_dummy(locale="ja")[0],
        ),
        (
            "invoice",
            lambda: [
                {
                    "text": "請求書: ベンダー=Acme; 日付=2024-12-31; 金額=1234.56; 通貨=USD",
                    "vendor": "Acme",
                    "date": "2024-12-31",
                    "amount": 1234.56,
                    "currency": "USD",
                }
            ],
            ["text"],
            lambda: F.invoice_dummy(locale="ja")[0],
        ),
        (
            "routed",
            lambda: [
                {
                    "query": "ID 42 のユーザーのメールアドレスは？",
                    "answer": "user42@example.com",
                    "preferred_source": "db",
                }
            ],
            ["query"],
            lambda: F.routed_sources_dummy(locale="ja")[0],
        ),
        (
            "image",
            lambda: [
                {
                    "image": dspy.Image(url="https://example.com/img.jpg"),
                    "keywords": ["犬", "屋外"],
                }
            ],
            ["image"],
            lambda: F.image_caption_dummy(locale="ja")[0],
        ),
        (
            "langextract",
            lambda: [
                {
                    "text": "ROMEO. But soft!... Juliet is the sun.",
                    "targets": [
                        {"extraction_class": "character", "extraction_text": "ROMEO"}
                    ],
                }
            ],
            ["text"],
            lambda: F.langextract_dummy(locale="en")[0],
        ),
    ],
)
def test_to_examples_compatibility_with_factory(name, make_rows, input_keys, factory_builder):
    """
    Given sample rows matching each demo schema
    When we convert with data_tools.to_examples
    Then keys and input fields are compatible with the dummy factory dataset
    """
    # Given
    rows = make_rows()
    factory_examples = factory_builder()
    assert isinstance(factory_examples, list) and len(factory_examples) > 0

    # When
    dt_examples = DT.to_examples(rows, input_keys=input_keys)

    # Then: compare first element key sets and input field names
    fe0 = factory_examples[0]
    de0 = dt_examples[0]
    assert set(fe0.keys()) == set(de0.keys()), f"{name}: key set mismatch"
    assert set(fe0.inputs().keys()) == set(de0.inputs().keys()), f"{name}: input keys mismatch"

    # Special check: image rows should carry dspy.Image objects unchanged
    if name == "image":
        assert isinstance(de0["image"], dspy.Image)
