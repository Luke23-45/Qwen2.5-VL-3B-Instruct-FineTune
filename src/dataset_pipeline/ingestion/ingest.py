from __future__ import annotations

from ..core.models import ProjectConfig
from ..sources import build_adapter


def download_sources(config: ProjectConfig) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for source_name, source_config in config.sources.items():
        if not source_config.enabled:
            continue
        adapter = build_adapter(source_config)
        statuses[source_name] = adapter.download_or_link()
    return statuses
