from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DatasetFormat = Literal["chat", "completion", "text"]


class DatasetFileReport(BaseModel):
    path: str
    format: DatasetFormat
    examples: int


class DatasetReport(BaseModel):
    data_path: str
    files: list[DatasetFileReport] = Field(default_factory=list)


def detect_record_format(record: dict) -> DatasetFormat:
    if isinstance(record.get("messages"), list):
        return "chat"
    if "prompt" in record and "completion" in record:
        return "completion"
    if isinstance(record.get("text"), str):
        return "text"
    raise ValueError("Expected a chat, completion, or text JSONL record.")


def validate_jsonl_file(path: str | Path) -> DatasetFileReport:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")
    if file_path.suffix != ".jsonl":
        raise ValueError(f"Dataset file must use .jsonl: {file_path}")

    detected_format: DatasetFormat | None = None
    examples = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_number} must be a JSON object.")
            record_format = detect_record_format(record)
            detected_format = detected_format or record_format
            examples += 1

    if examples == 0 or detected_format is None:
        raise ValueError(f"Dataset file has no examples: {file_path}")

    return DatasetFileReport(
        path=str(file_path),
        format=detected_format,
        examples=examples,
    )


def validate_dataset_dir(path: str | Path) -> DatasetReport:
    data_path = Path(path)
    if data_path.is_file():
        return DatasetReport(data_path=str(data_path), files=[validate_jsonl_file(data_path)])

    train_file = data_path / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"Expected train.jsonl in dataset directory: {data_path}")

    reports = [validate_jsonl_file(train_file)]
    for optional_name in ("valid.jsonl", "test.jsonl"):
        optional_file = data_path / optional_name
        if optional_file.exists():
            reports.append(validate_jsonl_file(optional_file))

    return DatasetReport(data_path=str(data_path), files=reports)
