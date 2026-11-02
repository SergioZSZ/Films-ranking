import io
import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd

from src.rt.predict_missing_scores import (
    assemble_output,
    build_model_texts,
    classify_rows,
    load_reviews,
    main,
    parse_args,
    reconstruct_scores,
    validate_model_artifact,
)


VALID_METADATA = {
    "status": "completed",
    "objective": "normalized_score - mean(review_type)",
    "input_template": (
        "Sentiment classification: {review_type}. Review: {review_content}"
    ),
    "postprocessing": "clip(mean(review_type) + predicted_residual, 0, 10)",
    "means": {"Fresh": 7.4, "Rotten": 3.2},
    "max_length": 256,
}


def create_model_artifact(root: Path, metadata: dict | None = None) -> Path:
    model_dir = root / "epac-roberta-residual-full"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "roberta",
                "num_labels": 1,
                "problem_type": "regression",
            }
        ),
        encoding="utf-8",
    )
    for name in ("model.safetensors", "tokenizer_config.json", "tokenizer.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "training_metadata.json").write_text(
        json.dumps(metadata or VALID_METADATA), encoding="utf-8"
    )
    return model_dir


class ModelArtifactContractTests(unittest.TestCase):
    def test_accepts_completed_residual_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = create_model_artifact(Path(tmp))

            metadata = validate_model_artifact(model_dir)

        self.assertEqual(metadata["means"], {"Fresh": 7.4, "Rotten": 3.2})
        self.assertEqual(metadata["max_length"], 256)

    def test_rejects_artifact_that_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = {**VALID_METADATA, "status": "running"}
            model_dir = create_model_artifact(Path(tmp), metadata)

            with self.assertRaisesRegex(ValueError, "status.*completed"):
                validate_model_artifact(model_dir)

    def test_rejects_artifact_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = create_model_artifact(Path(tmp))
            (model_dir / "model.safetensors").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "weights"):
                validate_model_artifact(model_dir)

    def test_rejects_model_config_that_is_not_single_output_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = create_model_artifact(Path(tmp))
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "roberta",
                        "num_labels": 2,
                        "problem_type": "single_label_classification",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "config.json.*num_labels"):
                validate_model_artifact(model_dir)


class InputContractTests(unittest.TestCase):
    def test_rejects_normalized_scores_outside_zero_to_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "reviews.csv"
            pd.DataFrame(
                {
                    "review_score": ["11/10"],
                    "normalized_score": ["11"],
                    "review_content": ["Too high"],
                    "review_type": ["Fresh"],
                }
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, r"normalized_score.*\[0, 10\]"):
                load_reviews(csv_path)


def review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_score": ["4/5", "4", "", "", "", ""],
            "normalized_score": ["8.5", "", "", "", "", ""],
            "review_content": [
                "Already scored",
                "Ambiguous scale",
                "   ",
                "Has text",
                "Excellent film",
                "Very poor film",
            ],
            "review_type": ["Fresh", "Fresh", "Fresh", "Mixed", " fresh ", "Rotten"],
        }
    )


class RowClassificationTests(unittest.TestCase):
    def test_classifies_every_row_without_overwriting_explicit_scores(self):
        sources, eligible = classify_rows(review_frame())

        self.assertEqual(
            sources.tolist(),
            [
                "normalized_original",
                "unavailable_unparseable",
                "unavailable_missing_text",
                "unavailable_invalid_review_type",
                "pending_prediction_limit",
                "pending_prediction_limit",
            ],
        )
        self.assertEqual(np.flatnonzero(eligible.to_numpy()).tolist(), [4, 5])

    def test_builds_the_exact_training_text(self):
        texts = build_model_texts(review_frame(), np.array([4]))

        self.assertEqual(
            texts,
            ["Sentiment classification: Fresh. Review: Excellent film"],
        )


class ResidualReconstructionTests(unittest.TestCase):
    def test_adds_review_type_means_and_clips_to_score_range(self):
        scores = reconstruct_scores(
            residuals=np.array([0.5, -5.0, 5.0]),
            review_types=["Fresh", "Rotten", "Fresh"],
            means={"Fresh": 7.4, "Rotten": 3.2},
        )

        np.testing.assert_allclose(scores, np.array([7.9, 0.0, 10.0]))

    def test_rejects_non_finite_model_outputs(self):
        with self.assertRaisesRegex(ValueError, "no finitas"):
            reconstruct_scores(
                residuals=np.array([np.nan]),
                review_types=["Fresh"],
                means={"Fresh": 7.4, "Rotten": 3.2},
            )


class OutputAssemblyTests(unittest.TestCase):
    def test_adds_predictions_final_scores_and_provenance(self):
        output = assemble_output(
            reviews=review_frame(),
            predicted_indices=np.array([4]),
            predictions=np.array([7.92346]),
            model_name="epac-roberta-residual-full",
            metadata_identifier="training_metadata.json",
        )

        self.assertEqual(output.loc[0, "final_score"], "8.5")
        self.assertEqual(output.loc[1, "final_score"], "")
        self.assertEqual(output.loc[4, "predicted_score"], "7.9235")
        self.assertEqual(output.loc[4, "final_score"], "7.9235")
        self.assertEqual(output.loc[4, "score_source"], "roberta_prediction")
        self.assertEqual(output.loc[5, "score_source"], "pending_prediction_limit")
        self.assertEqual(output.loc[4, "prediction_model"], "epac-roberta-residual-full")
        self.assertEqual(output.loc[4, "prediction_metadata"], "training_metadata.json")
        self.assertEqual(output.loc[5, "prediction_model"], "")


class CommandLineTests(unittest.TestCase):
    def test_uses_repository_relative_full_model_defaults(self):
        args = parse_args([])

        self.assertEqual(
            args.input.name,
            "rotten_tomatoes_critic_reviews_clean.csv",
        )
        self.assertEqual(args.input.parent.name, "rottentomatoes")
        self.assertEqual(args.input.parent.parent.name, "processed_datasets")
        self.assertEqual(args.model.name, "epac-roberta-residual-full")
        self.assertEqual(
            args.output.name,
            "rotten_tomatoes_critic_reviews_scored.csv",
        )
        self.assertEqual(args.device, "auto")

    def test_parses_validation_only_without_loading_a_model(self):
        args = parse_args(["--validate-only", "--limit", "100", "--device", "cpu"])

        self.assertTrue(args.validate_only)
        self.assertEqual(args.limit, 100)
        self.assertEqual(args.device, "cpu")

    def test_rejects_non_positive_batch_size(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--batch-size", "0"])

    def test_rejects_non_positive_limit(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--limit", "-1"])

    def test_refuses_to_overwrite_the_input_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "reviews.csv"
            original = "review_score,normalized_score,review_content,review_type\n"
            input_csv.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "entrada.*salida"):
                main(
                    [
                        "--input",
                        str(input_csv),
                        "--output",
                        str(input_csv),
                        "--model",
                        str(Path(tmp) / "missing-model"),
                    ]
                )

            self.assertEqual(input_csv.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
