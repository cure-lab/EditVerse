#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Ref     :   https://github.com/yuvalkirstain/PickScore
"""
import logging

import torch
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class PickScoreVideoQuality:
    def __init__(self, device: torch.device):
        model_repo = "yuvalkirstain/PickScore_v1"
        processor_repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

        self.device = device
        logger.debug("Initializing CLIP processor and model...")

        self.preprocessor = CLIPProcessor.from_pretrained(processor_repo)
        self.model = CLIPModel.from_pretrained(model_repo).to(device)
        self.model.eval()

        logger.debug("Successfully loaded %s", self.model.__class__.__name__)

    def preprocess(self, video, target_prompt):
        # Prepare text input
        text_inputs = self.preprocessor(
            text=target_prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.device)

        # Prepare image inputs with resizing if necessary
        image_inputs = []
        for frame in video:
            width, height = frame.size
            largest_side = max(width, height)

            if largest_side > 672:
                scale_ratio = 672.0 / largest_side
                new_dims = (int(width * scale_ratio), int(height * scale_ratio))
                frame = frame.resize(new_dims)

            proc_frame = self.preprocessor(
                images=frame,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(self.device)

            image_inputs.append(proc_frame)

        return text_inputs, image_inputs

    @torch.no_grad()
    def evaluate(self, video, target_prompt) -> float:
        text_inputs, image_inputs = self.preprocess(video, target_prompt)

        # Normalize text embeddings
        text_features = self.model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        frame_scores = []
        for batch in image_inputs:
            image_features = self.model.get_image_features(**batch)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

            logits = text_features @ image_features.T
            scaled_score = (self.model.logit_scale.exp() * logits).cpu().squeeze().item()
            frame_scores.append(scaled_score)

        return float(sum(frame_scores) / len(frame_scores))