from __future__ import annotations

from evaluations.validations.data import parse_json_object


PARSE_ERROR_LABEL = "__parse_error__"


def predicted_canonical_label(text: str) -> str:
    parsed = parse_json_object(text)
    if parsed and parsed.get("canonical_label"):
        return str(parsed["canonical_label"])
    return PARSE_ERROR_LABEL
