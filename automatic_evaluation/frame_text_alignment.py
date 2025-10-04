#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Ref:
    https://github.com/openai/CLIP
    https://github.com/mlfoundations/open_clip
    https://huggingface.co/docs/transformers/model_doc/clip#clip
"""

import logging
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor


logger = logging.getLogger(__name__)


class FrameTextAlignment:
    def __init__(self, device: torch.device):
        model_id = "openai/clip-vit-large-patch14"
        self.device = device

        logger.debug("Initializing CLIP with weights: %s", model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.debug("Successfully loaded %s", self.model.__class__.__name__)

    def preprocess(self, video, target_prompt):
        # Prepare text batch
        text_batch = self.processor(
            text=target_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        # Process each frame individually
        frame_batches = [
            self.processor(
                images=frame,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)
            for frame in video
        ]

        return text_batch, frame_batches

    @torch.no_grad()
    def evaluate(self, video, target_prompt) -> float:
        text_inputs, image_batches = self.preprocess(video, target_prompt)

        # Normalize text embedding
        text_features = self.model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # Compute similarities
        similarity_scores = []
        for batch in image_batches:
            img_features = self.model.get_image_features(**batch)
            img_features = torch.nn.functional.normalize(img_features, dim=-1)

            logits = self.model.logit_scale.exp() * text_features @ img_features.T
            similarity_scores.append(logits.cpu().item())

        # Return average similarity
        return float(sum(similarity_scores) / len(similarity_scores))
