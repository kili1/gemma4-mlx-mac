from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gemma4_mlx_mac.media import prepare_multimodal_inputs


def test_prepare_multimodal_inputs_decodes_base64_image_and_audio_and_cleans_up() -> None:
    png_bytes = b"\x89PNG\r\n\x1a\nfakepng"
    wav_bytes = b"RIFF0000WAVEfmt "
    img_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
    audio_b64 = base64.b64encode(wav_bytes).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": "Analyze both."},
            ],
        }
    ]

    with prepare_multimodal_inputs(messages) as batch:
        assert batch.has_media is True
        assert len(batch.images) == 1
        assert len(batch.audios) == 1
        img_path = Path(batch.images[0])
        audio_path = Path(batch.audios[0])
        assert img_path.read_bytes() == png_bytes
        assert audio_path.read_bytes() == wav_bytes
        assert batch.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "audio"},
                    {"type": "text", "text": "Analyze both."},
                ],
            }
        ]

    # Temp files are cleaned up automatically on context exit.
    assert not img_path.exists()
    assert not audio_path.exists()


def test_prepare_multimodal_inputs_attaches_cli_media_and_blocks_remote_urls(
    tmp_path: Path,
) -> None:
    local_img = tmp_path / "diagram.png"
    local_img.write_bytes(b"png")

    with prepare_multimodal_inputs(
        [{"role": "user", "content": "Explain this diagram."}],
        extra_images=[str(local_img)],
    ) as batch:
        assert batch.images == [str(local_img)]
        assert batch.messages[0]["content"] == [
            {"type": "image"},
            {"type": "text", "text": "Explain this diagram."},
        ]

    with pytest.raises(ValueError, match="local-only mode"):
        prepare_multimodal_inputs(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
                    ],
                }
            ]
        )


def test_prepare_multimodal_inputs_rejects_missing_local_file() -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_multimodal_inputs(
            [{"role": "user", "content": "Hi"}],
            extra_images=["/does/not/exist/photo.png"],
        )
