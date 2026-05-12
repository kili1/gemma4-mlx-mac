from pathlib import Path

import pytest

from gemma4_mlx_mac.datasets import (
    SyntheticDatasetRequest,
    create_synthetic_dataset,
    validate_dataset_dir,
    validate_jsonl_file,
)


def test_validate_dataset_dir_accepts_examples() -> None:
    report = validate_dataset_dir(Path("examples/data"))

    assert report.files[0].examples == 2
    assert report.files[0].format == "chat"


def test_validate_jsonl_file_rejects_bad_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"question":"hello"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected"):
        validate_jsonl_file(dataset)


def test_create_synthetic_dataset_writes_valid_train_file(tmp_path: Path) -> None:
    result = create_synthetic_dataset(
        SyntheticDatasetRequest(
            topic="MLX fine tuning",
            examples=3,
            output_dir=str(tmp_path / "synthetic"),
            format="chat",
        )
    )

    assert result.examples == 3
    assert result.report.files[0].examples == 3
    assert result.report.files[0].format == "chat"
