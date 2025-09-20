import json
from pathlib import Path

import pytest

import dspy
from real_world import data_tools as DT


def _assert_jsonl_file(path: Path):
    assert path.exists(), f"Missing file: {path}"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0, f"Empty JSONL: {path}"
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert isinstance(obj, dict), f"Line {i} not a JSON object"


@pytest.mark.parametrize(
    "name, make_rows, builder, loader, expected_input_keys, expected_keys",
    [
        (
            "qa",
            lambda: [
                {"question": "空の色は何色ですか？", "answer": "青"},
                {"question": "バナナの色は何色ですか？", "answer": "黄色"},
            ],
            DT.prepare_qa_from_jsonl,
            DT.load_qa_examples_from_jsonl,
            {"question"},
            {"question", "answer"},
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
                },
                {
                    "text": "請求書: ベンダー=Tokyo; 日付=2024-12-30; 金額=7890; 通貨=¥",
                    "vendor": "Tokyo",
                    "date": "2024-12-31",
                    "amount": "¥7,890",
                    "currency": "JPY",
                },
            ],
            DT.prepare_invoice_from_jsonl,
            DT.load_invoice_examples_from_jsonl,
            {"text"},
            {"text", "vendor", "date", "amount", "currency"},
        ),
        (
            "routed",
            lambda: [
                {"query": "ID 42 のユーザーのメールアドレスは？", "answer": "user42@example.com", "preferred_source": "db"},
                {"query": "最新のポリシー更新を要約してください。", "answer": "Policy updated", "preferred_source": "rag"},
            ],
            DT.prepare_routed_from_jsonl,
            DT.load_routed_examples_from_jsonl,
            {"query"},
            {"query", "answer", "preferred_source"},
        ),
        (
            "image",
            lambda: [
                {"image_url": "https://example.com/1.jpg", "keywords": ["犬", "屋外"]},
                {"image_url": "https://example.com/2.jpg", "keywords": "犬,屋外"},
            ],
            DT.prepare_image_caption_from_jsonl,
            DT.load_image_caption_examples_from_jsonl,
            {"image"},
            {"image", "keywords"},
        ),
        (
            "langextract",
            lambda: [
                {"text": "ROMEO. But soft!", "targets": [{"extraction_class": "character", "extraction_text": "ROMEO"}]},
                {"text": "Juliet is the sun.", "targets": [{"extraction_class": "relationship", "extraction_text": "Juliet is the sun"}]},
            ],
            DT.prepare_langextract_from_jsonl,
            DT.load_langextract_examples_from_jsonl,
            {"text"},
            {"text", "targets"},
        ),
    ],
)
def test_prepare_and_load_jsonl_roundtrip(tmp_path, name, make_rows, builder, loader, expected_input_keys, expected_keys):
    """
    Given domain-appropriate rows saved as JSONL
    When we prepare train/val JSONLs with data_tools and load them back as Examples
    Then files exist, lines are valid JSON, and Examples have expected keys and inputs
    """
    in_jsonl = tmp_path / f"{name}_input.jsonl"
    DT.save_jsonl(in_jsonl, make_rows())

    out_train = tmp_path / f"{name}_train.jsonl"
    out_val = tmp_path / f"{name}_val.jsonl"

    # Build train/val JSONLs and Examples
    train_exs, val_exs = builder(in_jsonl, out_train, out_val, val_ratio=0.5, seed=7)

    # Files should be present and valid JSONL
    _assert_jsonl_file(out_train)
    _assert_jsonl_file(out_val)

    # Load through domain loaders
    loaded_train = loader(out_train)
    loaded_val = loader(out_val)

    assert isinstance(train_exs, list) and isinstance(val_exs, list)
    assert isinstance(loaded_train, list) and isinstance(loaded_val, list)

    # There should be at least one example split between train/val
    assert len(train_exs) + len(val_exs) >= 1
    assert len(loaded_train) + len(loaded_val) >= 1

    # Compare schema on a sample if available
    sample_lists = [l for l in [train_exs, val_exs, loaded_train, loaded_val] if len(l) > 0]
    assert sample_lists, "No examples produced"
    sample = sample_lists[0][0]
    assert set(sample.keys()) == expected_keys
    assert set(sample.inputs().keys()) == expected_input_keys

    # For image tasks, ensure image type is dspy.Image
    if name == "image":
        any_list = sample_lists[0]
        assert isinstance(any_list[0]["image"], dspy.Image)

