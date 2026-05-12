from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DatasetFormat = Literal["chat", "completion", "text"]


class DatasetFileReport(BaseModel):
    path: str
    format: DatasetFormat
    examples: int


class DatasetReport(BaseModel):
    data_path: str
    files: list[DatasetFileReport] = Field(default_factory=list)


class SyntheticDatasetRequest(BaseModel):
    topic: str = "local Apple Silicon AI"
    examples: int = Field(default=24, ge=1, le=500)
    output_dir: str = "examples/synthetic"
    format: DatasetFormat = "chat"
    system_prompt: str = "You are a concise, helpful assistant."

    @field_validator("topic", "output_dir", "system_prompt")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value is required.")
        return cleaned


class SyntheticDatasetResult(BaseModel):
    data_path: str
    train_path: str
    format: DatasetFormat
    examples: int
    report: DatasetReport


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


def create_synthetic_dataset(request: SyntheticDatasetRequest) -> SyntheticDatasetResult:
    output_dir = Path(request.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for index in range(request.examples):
            handle.write(f"{json.dumps(_synthetic_record(request, index))}\n")

    report = validate_dataset_dir(output_dir)
    return SyntheticDatasetResult(
        data_path=str(output_dir),
        train_path=str(train_path),
        format=request.format,
        examples=request.examples,
        report=report,
    )


def _synthetic_record(request: SyntheticDatasetRequest, index: int) -> dict:
    prompt, answer = _synthetic_pair(request.topic, index)
    if request.format == "chat":
        return {
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
    if request.format == "completion":
        return {"prompt": prompt, "completion": answer}
    return {"text": f"Topic: {request.topic}\nQuestion: {prompt}\nAnswer: {answer}"}


def _synthetic_pair(topic: str, index: int) -> tuple[str, str]:
    templates = [
        (
            "Explain {topic} to a beginner.",
            "{topic} is best understood by starting with the goal, the main constraints, "
            "and one small working example. Keep the setup simple, verify each step, and "
            "increase complexity only after the first result works.",
        ),
        (
            "List three practical tips for working with {topic}.",
            "First, define the exact outcome you want. Second, keep inputs and outputs easy "
            "to inspect. Third, record the settings that produced a good result so the "
            "workflow can be repeated.",
        ),
        (
            "What should a user check before starting a {topic} workflow?",
            "The user should check available memory, confirm dependencies are installed, "
            "choose a small initial dataset, and run a short validation pass before a longer job.",
        ),
        (
            "Write a concise troubleshooting answer for {topic}.",
            "Start by reproducing the issue with the smallest example. Check paths, versions, "
            "and input format. Then rerun with clear logs so the failing step is visible.",
        ),
        (
            "Summarize the safest way to iterate on {topic}.",
            "Use short runs, review outputs often, keep the original data unchanged, and save "
            "each useful configuration with a clear name.",
        ),
    ]
    prompt_template, answer_template = templates[index % len(templates)]
    prompt = prompt_template.format(topic=topic)
    answer = answer_template.format(topic=topic)
    if index >= len(templates):
        answer = f"{answer} Example {index + 1} focuses on a slightly different usage scenario."
    return prompt, answer
