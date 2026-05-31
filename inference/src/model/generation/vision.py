from __future__ import annotations

from typing import Any


def collect_vision_inputs(messages_batch: list[list[dict]]) -> tuple[list[Any] | None, list[Any] | None]:
    from qwen_vl_utils import process_vision_info

    image_inputs: list[Any] = []
    video_inputs: list[Any] = []
    for messages in messages_batch:
        images, videos = process_vision_info(messages)
        image_inputs.extend(images or [])
        video_inputs.extend(videos or [])
    return image_inputs or None, video_inputs or None
