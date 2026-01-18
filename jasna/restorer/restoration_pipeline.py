from __future__ import annotations

import torch

from jasna.mosaic import Detections
from jasna.restorer.frames_restorer import FramesRestorer


class RestorationPipeline:
    def __init__(self, *, frame_restorer: FramesRestorer) -> None:
        self.frame_restorer = frame_restorer

    def restore(self, frames_uint8_bchw: torch.Tensor, detections: Detections) -> torch.Tensor:
        return self.frame_restorer.restore(frames_uint8_bchw, detections)

