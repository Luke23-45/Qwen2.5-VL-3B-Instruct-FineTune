from __future__ import annotations

from .adapters import (
    AnnotationFileAdapter,
    LocalFolderAdapter,
    ManualMirrorAdapter,
    MetadataFolderAdapter,
    URLArchiveAdapter,
)


ADAPTERS = {
    "local_folder": LocalFolderAdapter,
    "manual_mirror": ManualMirrorAdapter,
    "url_archive": URLArchiveAdapter,
    "metadata_folder": MetadataFolderAdapter,
    "annotation_file": AnnotationFileAdapter,
}


def build_adapter(config):
    adapter_cls = ADAPTERS.get(config.adapter)
    if adapter_cls is None:
        raise KeyError(f"Unsupported adapter: {config.adapter}")
    return adapter_cls(config)
