import argparse
from automatic_evaluation import EvaluatorWrapper


def main(args):
    # Announce setup
    print(">>> Preparing evaluation environment...")

    # Build evaluator with provided configuration
    evaluator = EvaluatorWrapper(
        gpt_api_key=args.gpt_api_key,
        metrics=args.metrics,
        test_json_path=args.test_json_path
    )

    # Display run info
    print(f"Running evaluation using metrics: {args.metrics}")
    print(f"Evaluating results from directory: {args.generate_results_dir}")

    # Perform evaluation
    evaluator.evaluate(
        generate_results_dir=args.generate_results_dir,
        output_csv=args.output_csv
    )

    # Wrap up
    print(f"✅ Done! Results written to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatic video evaluation runner with customizable metrics."
    )

    # Metrics configuration
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated list of metrics to include. "
            'Defaults to "all". Example: '
            "clip_temporal_consistency,dino_temporal_consistency,"
            "frame_text_alignment,video_text_alignment,"
            "pick_score_video_quality,editing_vlm_evaluation"
        ),
    )

    # File paths and directories
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="EditVerseBench/EditVerseBench/test.json",
        help="Path pointing to the test JSON specification."
    )
    parser.add_argument(
        "--generate_results_dir",
        type=str,
        default="EditVerseBench/EditVerse_Comparison_Results/EditVerse",
        help="Folder containing generated evaluation targets."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="EditVerse_eval.csv",
        help="Destination filename for evaluation results in CSV format."
    )

    # API Key
    parser.add_argument(
        "--gpt_api_key",
        type=str,
        required=True,
        help="API key for GPT access (required)."
    )

    # Execute
    arguments = parser.parse_args()
    main(arguments)
