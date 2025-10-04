#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import torch
import logging
import warnings
from typing import Dict, List, Optional, Union

import decord
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger(__name__)


class EvaluatorWrapper:
    _temporal_consistency = ["clip_temporal_consistency", "dino_temporal_consistency"]
    _text_alignment = ["frame_text_alignment", "video_text_alignment"]
    _video_quality = ["pick_score_video_quality"]
    _vlm_evaluation = ["editing_vlm_evaluation"]

    all_metrics = (
        _temporal_consistency
        + _text_alignment
        + _video_quality
        + _vlm_evaluation
    )

    def __init__(
        self,
        metrics: Union[str, List[str]] = "all",
        test_json_path: str = "benchmark/test.json",
        gpt_api_key: str = "",
        device: Optional[torch.device] = None,
    ):
        self.metrics = metrics
        self._validate_metrics()

        with open(test_json_path, "r") as f:
            self.test_json = json.load(f)

        self.gpt_api_key = gpt_api_key
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._load_evaluators()

    def _validate_metrics(self):
        """Ensure only supported metrics are selected."""
        if isinstance(self.metrics, str):
            self.metrics = (
                self.all_metrics if self.metrics == "all" else self.metrics.split(",")
            )

        unsupported = set(self.metrics) - set(self.all_metrics)
        if unsupported:
            warnings.warn(f"Unsupported metrics ignored: {unsupported}")
            self.metrics = [m for m in self.metrics if m not in unsupported]

        if not self.metrics:
            raise ValueError("No valid metrics specified!")

        logger.info("*" * 50)
        logger.info(f"Enabled metrics: {self.metrics}")
        logger.info("*" * 50)

    def _load_evaluators(self):
        """Dynamically import and initialize evaluators."""
        if "clip_temporal_consistency" in self.metrics:
            from automatic_evaluation.clip_temporal_consistency import (
                ClipTemporalConsistency,
            )

            self.clip_temporal_consistency_evaluator = ClipTemporalConsistency(
                self.device
            )

        if "dino_temporal_consistency" in self.metrics:
            from automatic_evaluation.dino_temporal_consistency import (
                DinoTemporalConsistency,
            )

            self.dino_temporal_consistency_evaluator = DinoTemporalConsistency(
                self.device
            )

        if "frame_text_alignment" in self.metrics:
            from automatic_evaluation.frame_text_alignment import FrameTextAlignment

            self.frame_text_alignment_evaluator = FrameTextAlignment(self.device)

        if "video_text_alignment" in self.metrics:
            from automatic_evaluation.video_text_alignment import VideoTextAlignment

            self.video_text_alignment_evaluator = VideoTextAlignment(self.device)

        if "pick_score_video_quality" in self.metrics:
            from automatic_evaluation.pick_score_video_quality import (
                PickScoreVideoQuality,
            )

            self.pick_score_video_quality_evaluator = PickScoreVideoQuality(self.device)

        if "editing_vlm_evaluation" in self.metrics:
            from automatic_evaluation.editing_vlm_evaluation import EditingVLMEvaluation

            self.editing_vlm_evaluation_evaluator = EditingVLMEvaluation(
                self.gpt_api_key
            )

    def _read_video_frames(self, video_path: str) -> List[Image.Image]:
        """Helper function to decode all frames from a video into a list of PIL Images."""
        frames = []
        vr = decord.VideoReader(video_path, num_threads=1)
        for idx in range(len(vr)):
            frames.append(Image.fromarray(vr[idx].asnumpy()))
        return frames

    def evaluate(
        self,
        generate_results_dir: str = "results/EditVerse_original",
        output_csv: str = "EditVerse_original.csv",
    ) -> Dict[str, Dict[str, List]]:
        """Run evaluation on generated and reference videos, save results to CSV."""
        results_df = pd.DataFrame(columns=["id"] + self.metrics + ["type"])

        for vid_id, meta in tqdm(self.test_json.items()):
            row_results = [vid_id]

            gen_path = os.path.join(generate_results_dir, vid_id, "generate.mp4")
            ref_path = os.path.join(generate_results_dir, vid_id, "video1.mp4")

            generated_frames = self._read_video_frames(gen_path)
            reference_frames = self._read_video_frames(ref_path)

            editing_prompt = meta["<text>"]
            target_prompt = meta["target_prompt"]

            for metric in self.metrics:
                logger.info(f"Evaluating {metric} ...")
                if metric == "clip_temporal_consistency":
                    row_results.append(
                        self.clip_temporal_consistency_evaluator.evaluate(
                            generated_frames
                        )
                    )
                elif metric == "dino_temporal_consistency":
                    row_results.append(
                        self.dino_temporal_consistency_evaluator.evaluate(
                            generated_frames
                        )
                    )
                elif metric == "frame_text_alignment":
                    row_results.append(
                        self.frame_text_alignment_evaluator.evaluate(
                            generated_frames, target_prompt
                        )
                    )
                elif metric == "video_text_alignment":
                    row_results.append(
                        self.video_text_alignment_evaluator.evaluate(
                            generated_frames, target_prompt
                        )
                    )
                elif metric == "pick_score_video_quality":
                    row_results.append(
                        self.pick_score_video_quality_evaluator.evaluate(
                            generated_frames, target_prompt
                        )
                    )
                elif metric == "editing_vlm_evaluation":
                    row_results.append(
                        self.editing_vlm_evaluation_evaluator.evaluate(
                            reference_frames, generated_frames, editing_prompt
                        )
                    )

            row_results.append(meta["type"])
            results_df.loc[len(results_df)] = row_results

            # continuously save progress
            results_df.to_csv(output_csv, index=False)

        logger.info(f"Evaluation finished. Writing results to {output_csv} ...")

        # compute averages for numeric metrics
        numeric_data = results_df.select_dtypes(include=np.number)
        averages = numeric_data.mean()
        results_df.loc["Average"] = pd.Series(averages, name="Average")
        results_df.loc["Average", "id"] = "Average"

        results_df.to_csv(output_csv, index=False)
        logger.info("Results saved successfully.")

        return results_df
