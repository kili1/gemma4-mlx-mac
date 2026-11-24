from pathlib import Path

import pytest

from gemma4_mlx_mac.datasets import (
    SyntheticDatasetRequest,
    create_synthetic_dataset,
    validate_dataset_dir,
    validate_jsonl_file,
)

EXAMPLES_DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_validate_dataset_dir_accepts_examples() -> None:
    report = validate_dataset_dir(EXAMPLES_DATA)

    assert report.files[0].examples == 2
    assert report.files[0].format == "chat"


def test_validate_jsonl_file_rejects_bad_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"question":"hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected"):
        validate_jsonl_file(dataset)


def test_validate_jsonl_file_rejects_mixed_formats(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n'
        '{"prompt":"q","completion":"a"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixes format"):
        validate_jsonl_file(dataset)


def test_validate_jsonl_file_rejects_malformed_messages(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages":[{"role":"invalid_role","content":"hi"}]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid role"):
        validate_jsonl_file(dataset)


def test_create_synthetic_dataset_writes_train_and_valid_files(tmp_path: Path) -> None:
    result = create_synthetic_dataset(
        SyntheticDatasetRequest(
            topic="MLX fine tuning",
            examples=3,
            output_dir=str(tmp_path / "synthetic"),
            format="chat",
        )
    )

    assert result.examples == 3
    assert result.valid_path is not None
    assert Path(result.valid_path).exists()
    assert result.report.files[0].examples == 3
    assert result.report.files[0].format == "chat"
    assert len(result.report.files) == 2


def test_create_synthetic_dataset_blocks_sensitive_directories() -> None:
    with pytest.raises(ValueError, match="sensitive directory"):
        create_synthetic_dataset(
            SyntheticDatasetRequest(
                topic="test",
                examples=1,
                output_dir="~/.ssh/gemma4-test",
            )
        )
