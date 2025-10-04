#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os

import torch

from automatic_evaluation.viclip import SimpleTokenizer, ViCLIP, clip_image_transform

logger = logging.getLogger(__name__)


class VideoTextAlignment:
    def __init__(
        self,
        device: torch.device,
        pretrained_tokenizer: str = None,
        pretrained_checkpoint: str = None,
    ):
        # Default paths for tokenizer and model checkpoint
        pretrained_tokenizer = pretrained_tokenizer or "automatic_evaluation/ckpt/bpe_simple_vocab_16e6.txt.gz"
        pretrained_checkpoint = pretrained_checkpoint or "automatic_evaluation/ckpt/ViClip-InternVid-10M-FLT.pth"

        # Ensure resources exist, otherwise fetch them
        if not os.path.isfile(pretrained_tokenizer):
            os.system(
                "wget https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz "
                "-P automatic_evaluation/ckpt"
            )
        if not os.path.isfile(pretrained_checkpoint):
            os.system(
                "wget https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth "
                "-P automatic_evaluation/ckpt"
            )

        self.device = device
        logger.debug(f"Initializing model from checkpoint: {pretrained_checkpoint}")

        # Build tokenizer and model
        tokenizer = SimpleTokenizer(bpe_path=pretrained_tokenizer)
        self.model = ViCLIP(tokenizer=tokenizer, pretrain=pretrained_checkpoint).to(self.device)
        self.model.eval()

        logger.debug(f"Successfully loaded {self.model.__class__.__name__}")

        # Standard transform for image preprocessing
        self.image_transform = clip_image_transform(224)

    def preprocess(self, video, target_prompt):
        # Sample 8 evenly spaced frames
        interval = len(video) / 8.0
        selected_frames = [video[int(round(i * interval))] for i in range(8)]

        # Apply image transformations
        processed_frames = [self.image_transform(frame) for frame in selected_frames]
        video_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        logger.debug(f"Prepared video tensor with shape: {video_tensor.shape}")

        return target_prompt, video_tensor

    @torch.no_grad()
    def evaluate(self, video, target_prompt) -> float:
        # Prepare inputs
        text_inputs, video_inputs = self.preprocess(video, target_prompt)

        # Encode and normalize text embeddings
        text_embs: torch.Tensor = self.model.encode_text(text_inputs).float()
        text_embs = torch.nn.functional.normalize(text_embs, dim=-1)
        logger.debug(f"Text embeddings: {text_embs.shape}")

        # Encode and normalize video embeddings
        video_embs: torch.Tensor = self.model.encode_vision(video_inputs, test=True).float()
        video_embs = torch.nn.functional.normalize(video_embs, dim=-1)
        logger.debug(f"Video embeddings: {video_embs.shape}")

        # Compute similarity score
        similarity = torch.matmul(text_embs, video_embs.T).cpu().squeeze().item()

        return similarity * 100
