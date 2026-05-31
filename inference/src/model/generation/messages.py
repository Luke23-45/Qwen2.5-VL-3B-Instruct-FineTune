from __future__ import annotations

from ..core.types import InferenceRequest


def qwen_messages(request: InferenceRequest) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(request.image)},
                {"type": "text", "text": request.prompt},
            ],
        }
    ]


def render_chat_texts(processor, requests: list[InferenceRequest]) -> tuple[list[str], list[list[dict]]]:
    messages = [qwen_messages(request) for request in requests]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    return texts, messages
