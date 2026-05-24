from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .base import gguf, logger
from .qwen import Qwen2Model

if TYPE_CHECKING:
    from torch import Tensor


class HailoModel(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.HAILO

    hef_path: Path | None = None

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # No host weight tensors are needed as everything is in the .hef (Hailo Executable Format).
        return {}

    def prepare_tensors(self):
        super().prepare_tensors()

        if self.hef_path is None:
            logger.warning("hailo: no --hef path provided; HEF tensor will be omitted")
            return

        hef_bytes = self.hef_path.read_bytes()
        if len(hef_bytes) == 0:
            logger.warning(f"hailo: HEF file {self.hef_path} is empty; tensor will be omitted")
            return

        hef_data = np.frombuffer(hef_bytes, dtype=np.uint8)
        logger.info(f"hailo: embedding HEF binary from {self.hef_path} ({len(hef_data)} bytes)")
        self.gguf_writer.add_tensor("hailo.hef_data", hef_data, raw_dtype=gguf.GGMLQuantizationType.I8)
