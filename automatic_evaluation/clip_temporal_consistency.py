#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
References:
    - https://github.com/openai/CLIP
    - https://github.com/mlfoundations/open_clip
    - https://huggingface.co/docs/transformers/model_doc/clip#clip
"""

import logging
from pathlib import Path

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


logger = logging.getLogger(__name__)


class ClipTemporalConsistency:
    def __init__(self, device: torch.device):
        model_id = "openai/clip-vit-large-patch14"
        self.device = device

        logger.debug(f"Initializing model {model_id}")
        self.preprocessor = CLIPImageProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.debug(f"{self.model.__class__.__name__} successfully initialized")

    def preprocess(self, video):
        # Process all frames in a list comprehension for cleaner flow
        return [
            self.preprocessor(frame, return_tensors="pt").pixel_values
            for frame in video
        ]

    @torch.no_grad()
    def evaluate(self, video) -> float:
        # Extract preprocessed frames
        frames = self.preprocess(video)

        # Prepare container for similarity scores
        scores = []
        prev_feat = None

        for idx, frame_tensor in enumerate(frames):
            # Send frame to device
            frame_tensor = frame_tensor.to(self.device)

            # Extract normalized features
            features = self.model(pixel_values=frame_tensor).pooler_output
            features = torch.nn.functional.normalize(features, dim=-1)

            # Compute similarity only if a previous frame exists
            if prev_feat is not None:
                sim = torch.matmul(features, prev_feat.T).cpu().squeeze().item()
                scores.append(max(0.0, sim))

            prev_feat = features

        # Handle empty input gracefully
        return (sum(scores) / len(scores)) if scores else 0.0
