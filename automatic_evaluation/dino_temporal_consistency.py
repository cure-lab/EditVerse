#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
References:
    https://github.com/facebookresearch/dinov2
    https://huggingface.co/docs/transformers/model_doc/dinov2#dinov2
"""
import logging
from typing import List

import torch
from transformers import BitImageProcessor, Dinov2Model

logger = logging.getLogger(__name__)


class DinoTemporalConsistency:
    def __init__(self, device: torch.device):
        model_id = "facebook/dinov2-base"
        self.device = device

        logger.debug("Initializing model %s", model_id)
        self.preprocessor: BitImageProcessor = BitImageProcessor.from_pretrained(model_id)
        self.model: Dinov2Model = Dinov2Model.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.debug("Successfully initialized %s", self.model.__class__.__name__)

    def preprocess(self, video) -> List[torch.Tensor]:
        """Convert raw frames into model-ready tensors."""
        processed = [
            self.preprocessor(frame, return_tensors="pt").pixel_values
            for frame in video
        ]
        return processed

    @torch.no_grad()
    def evaluate(self, video) -> float:
        """Measure temporal feature similarity between consecutive frames."""
        frames = self.preprocess(video)
        if len(frames) < 2:
            return 0.0

        similarities: List[float] = []
        prev_feat = None

        for idx, tensor in enumerate(frames):
            tensor = tensor.to(self.device)
            embedding: torch.Tensor = self.model(pixel_values=tensor).pooler_output
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            if prev_feat is not None:
                score = torch.matmul(embedding, prev_feat.T).squeeze().item()
                similarities.append(max(0.0, score))

            prev_feat = embedding

        return float(sum(similarities) / len(similarities))
