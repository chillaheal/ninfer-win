"""Qwen3.8-Flash-Next NVFP4 artifact conversion target.

Unlike ``qwen3_8_27b`` this target quantizes the official BF16 checkpoint on
the fly (there is no official NVFP4 artifact to repack).  The PLE n-gram
embedding and its three 63-bit I64 config tensors are excluded from the
``.ninfer`` artifact and stored in a separate ``.ngram`` store.
"""

from .inventory_nvfp4 import (
    FORMAT_COUNTS,
    LAYOUT_COUNTS,
    MODEL_ID,
    OBJECT_SPECS,
    RESOURCE_SPECS,
    TARGET_KEY,
    TENSOR_SPECS,
)

__all__ = [
    "FORMAT_COUNTS",
    "LAYOUT_COUNTS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "TENSOR_SPECS",
]
