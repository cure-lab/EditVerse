#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import base64
import cv2
import torch
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

class EditingVLMEvaluation:
    def __init__(self, gpt_api_key):
        self.client = OpenAI(api_key=gpt_api_key)
        self.sample_frames = 3

    def get_base64(self, frame):
        success, buf = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Frame encoding failed")
        return base64.b64encode(buf).decode("utf-8")

    @torch.no_grad()
    def evaluate(self, source_video, target_video, editing_prompt) -> float:
        total_scores = []
        step = max(1, len(target_video) // self.sample_frames)
        frame_indices = range(0, len(target_video), step)

        for idx in frame_indices:
            before_img = self.get_base64(np.asarray(source_video[idx]))
            after_img = self.get_base64(np.asarray(target_video[idx]))

            # Keep retrying until we get a valid score
            while True:
                try:
                    # First request: ask for evaluation details
                    evaluation = self.client.responses.create(
                        model="gpt-4o-2024-11-20",
                        temperature=0,
                        input=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": self._build_instruction(editing_prompt)
                                },
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{before_img}"},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{after_img}"}
                            ]
                        }]
                    )

                    # Second request: extract only the overall score
                    score_response = self.client.responses.create(
                        model="gpt-4o-2024-11-20",
                        temperature=0,
                        input=[{
                            "role": "user",
                            "content": [{
                                "type": "input_text",
                                "text": (
                                    f"Please output the overall score mentioned in this sentence. "
                                    f"Only output the overall score number. Sentence: {evaluation.output_text}"
                                )
                            }]
                        }]
                    )

                    score_text = score_response.output_text.strip()
                    if score_text.isdigit():
                        total_scores.append(int(score_text))
                        break
                except Exception as err:
                    logger.warning(f"Retrying due to error: {err}")

        return float(sum(total_scores)) / len(total_scores)

    def _build_instruction(self, prompt: str) -> str:
        return (
            "You are a meticulous video editing quality evaluator. Your task is to assess a video edit "
            "by comparing the original and edited images with respect to a given prompt.\n"
            f"Editing Prompt:\n{prompt}\n"
            "Instructions:\nEvaluate the edit across three criteria (0-3 each) and provide justifications. "
            "At the end, include the Total Score.\n\n"
            "1. Prompt Following (0-3): How well does the edit follow the given prompt?\n"
            "2. Edit Quality (0-3): Visual realism, seamlessness, and lack of artifacts.\n"
            "3. Background Consistency (0-3): Preservation of unedited areas.\n\n"
            "Format your response as follows:\n"
            "Prompt Following: [score] - [justification]\n"
            "Edit Quality: [score] - [justification]\n"
            "Background Consistency: [score] - [justification]\n"
            "Total Score: [sum]\n"
        )
