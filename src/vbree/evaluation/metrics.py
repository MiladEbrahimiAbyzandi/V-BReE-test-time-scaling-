import json

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_cols}")


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def accuracy_score(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    dataset_id_col: str = "dataset_id",
    ground_truth_col: str = "answer",
):
    required_result_cols = {"id", "chosen_response", "selected_choice"}
    _require_columns(results, required_result_cols, "Results DataFrame")

    if dataset_id_col not in ground_truth.columns or ground_truth_col not in ground_truth.columns:
        raise ValueError(f"Define dataset_id_column and ground_truth_col in the function according to your dataset.")

    # Ensure one ground-truth row per question id to avoid ambiguous scoring.
    gt = ground_truth[[dataset_id_col, ground_truth_col]].dropna()
    if gt[dataset_id_col].duplicated().any():
        raise ValueError(f"Ground truth has duplicate IDs in '{dataset_id_col}'.")

    gt_map = gt.set_index(dataset_id_col)[ground_truth_col]

    correct = 0
    total = 0

    for qid, group in results.groupby("id"):
        chosen = group[group["chosen_response"] == True]  # noqa: E712
        if chosen.empty or qid not in gt_map.index:
            continue

        predicted = str(chosen.iloc[-1]["selected_choice"]).strip().upper()
        gold = str(gt_map.loc[qid]).strip().upper()
        if predicted == gold:
            correct += 1
        total += 1

    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
    }


def _build_judge_prompt(
    question: str,
    cot_ground_truth: str,
    start_answer: str,
    end_answer: str,
) -> str:
    return f"""You are an expert evaluator assessing reasoning quality.

    Evaluate Answer A and Answer B against the reference reasoning.

    Question:
    {question}

    Reference Reasoning:
    {cot_ground_truth}

    Answer A:
    {start_answer}

    Answer B:
    {end_answer}

    Instructions:
    - Score Answer A from 0 to 100 based on logical correctness and alignment with the reference reasoning.
    - Score Answer B from 0 to 100 based on logical correctness and alignment with the reference reasoning.
    - Provide exactly one sentence of feedback for Answer A.
    - Provide exactly one sentence of feedback for Answer B.
    - Set "improved" to true only if score_b > score_a; otherwise false.

    Scoring guide:
    - 0-40: incorrect logic or wrong conclusion
    - 41-70: partially correct logic with gaps or mistakes
    - 71-100: logically sound reasoning with correct conclusion

    Important:
    - Score each answer independently.
    - Focus on reasoning quality, not wording similarity.
    - Do not explain your process.
    - Do not restate the question.
    - Do not include any text before or after the JSON.
    - Output must be valid JSON only.

    Required output format:
    {{
    "score_a": 0,
    "score_b": 0,
    "feedback_a": "One sentence.",
    "feedback_b": "One sentence.",
    "improved": false
    }}
    """


def _build_response_format_judge():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "JudgeResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score_a": {"type": "number"},
                    "score_b": {"type": "number"},
                    "feedback_a": {"type": "string"},
                    "feedback_b": {"type": "string"},
                    "improved": {"type": "boolean"},
                },
                "required": [
                    "score_a",
                    "score_b",
                    "feedback_a",
                    "feedback_b",
                    "improved",
                ],
                "additionalProperties": False,
            },
        },
    }


def reasoning_analysis(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    judge_provider,
    n_models: int,
    cot_content: str = "cot_content",
    dataset_id_col: str = "dataset_id",
    verbose: bool = False,
) -> dict:
    _require_columns(
        results,
        {"id", "iteration", "question", "updated_answer", "chosen_response"},
        "Results DataFrame",
    )
    _require_columns(
        ground_truth,
        {dataset_id_col, cot_content},
        "Ground truth DataFrame",
    )
    if n_models <= 1:
        raise ValueError("n_models must be >= 2.")

    scores_a = []
    scores_b = []
    improved_flags = []

    gt_map = ground_truth.dropna(subset=[dataset_id_col, cot_content]).set_index(dataset_id_col)[cot_content]

    for qid, group in results.groupby("id"):
        group = group.sort_values("iteration")

        # Get ground truth for this question.
        if qid not in gt_map.index:
            continue
        cot = str(gt_map.loc[qid])
        question = str(group.iloc[0]["question"])

        # Starting point: answer at iteration (N-1), or last available if run ended early.
        start_idx = min(max(n_models - 1, 0), len(group) - 1)
        start_answer = str(group.iloc[start_idx]["updated_answer"])

        # Ending point: chosen response answer.
        chosen_rows = group[group["chosen_response"] == True]  # noqa: E712
        if chosen_rows.empty:
            continue
        end_answer = str(chosen_rows.iloc[-1]["updated_answer"])

        prompt = _build_judge_prompt(
            question=question,
            cot_ground_truth=cot,
            start_answer=start_answer,
            end_answer=end_answer,
        )

        response_format = _build_response_format_judge()
        raw = judge_provider.generate(prompt, response_format=response_format)

        # if verbose:

        #     from pathlib import Path
        #     project_root = Path(__file__).parent.parent.parent.parent
        #     output_path = project_root / "runs" / "debug_judge_raw.txt"
        #     # save raw response to a text file to inspect:
        #     with open(output_path, "a", encoding='utf-8') as f:
        #         f.write(f"Question ID: {qid}\n")
        #         f.write(f"RAW: {raw}\n")
        #         f.write("-" * 50 + "\n")

        try:
            result = json.loads(raw)
            score_a = float(result["score_a"])
            score_b = float(result["score_b"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            if verbose:
                print(f"Warning: Failed to parse judge response for question ID {qid}: {e}")
                print(f"Raw response was: {raw}")
                
            continue

        scores_a.append(score_a)
        scores_b.append(score_b)
        improved_flags.append(score_b > score_a)

    improvements = [b - a for a, b in zip(scores_a, scores_b)]
    return {
        "avg_start_score": _safe_mean(scores_a),
        "avg_end_score": _safe_mean(scores_b),
        "avg_improvement": _safe_mean(improvements),
        "pct_improved": (sum(improved_flags) / len(improved_flags)) if improved_flags else float("nan"),
        "judged_questions": len(scores_a),
    }


def confidence_analysis(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    dataset_id_col: str = "dataset_id",
    ground_truth_col: str = "answer",
) -> dict:
    _require_columns(
        results,
        {"id", "chosen_response", "selected_choice", "confidence_score"},
        "Results DataFrame",
    )
    _require_columns(
        ground_truth,
        {dataset_id_col, ground_truth_col},
        "Ground truth DataFrame",
    )

    confidence_correct = []
    confidence_wrong = []

    gt = ground_truth[[dataset_id_col, ground_truth_col]].dropna()
    if gt[dataset_id_col].duplicated().any():
        raise ValueError(f"Ground truth has duplicate IDs in '{dataset_id_col}'.")
    gt_map = gt.set_index(dataset_id_col)[ground_truth_col]

    for qid, group in results.groupby("id"):
        chosen = group[group["chosen_response"] == True]  # noqa: E712
        if chosen.empty or qid not in gt_map.index:
            continue

        confidence = chosen.iloc[-1]["confidence_score"]
        if pd.isna(confidence):
            continue

        predicted = str(chosen.iloc[-1]["selected_choice"]).strip().upper()
        correct_answer = str(gt_map.loc[qid]).strip().upper()

        if predicted == correct_answer:
            confidence_correct.append(float(confidence))
        else:
            confidence_wrong.append(float(confidence))

    mean_correct = _safe_mean(confidence_correct)
    mean_wrong = _safe_mean(confidence_wrong)

    return {
        "mean_confidence_correct": mean_correct,
        "mean_confidence_wrong": mean_wrong,
        "gap": mean_correct - mean_wrong,
        "valid_questions": len(confidence_correct) + len(confidence_wrong),
    }


def efficiency(results: pd.DataFrame) -> dict:
    _require_columns(results, {"id", "iteration", "chosen_response"}, "Results DataFrame")

    iterations_per_question = []
    chosen_answer_iters = []
    extra_repetitions = []

    for _, group in results.groupby("id"):
        group = group.sort_values("iteration")
        iterations_per_question.append(len(group))

        chosen_rows = group[group["chosen_response"] == True] 
        if chosen_rows.empty:
            continue

        chosen_answer_iter = int(chosen_rows.iloc[-1]["iteration"])
        chosen_answer_iters.append(chosen_answer_iter)

        diff = len(group) - chosen_answer_iter - 1
        extra_repetitions.append(diff)

    if not iterations_per_question:
        return {
            "avg_iterations": float("nan"),
            "min_iterations": float("nan"),
            "max_iterations": float("nan"),
            "avg_chosen_answer_iter": float("nan"),
            "min_chosen_answer_iter": float("nan"),
            "max_chosen_answer_iter": float("nan"),
            "avg_extra_repetitions": float("nan"),
            "min_extra_repetitions": float("nan"),
            "max_extra_repetitions": float("nan"),
            "valid_questions": 0,
        }

    return {
        "avg_iterations": float(np.mean(iterations_per_question)),
        "min_iterations": int(np.min(iterations_per_question)),
        "max_iterations": int(np.max(iterations_per_question)),
        "avg_chosen_answer_iter": _safe_mean(chosen_answer_iters),
        "min_chosen_answer_iter": int(np.min(chosen_answer_iters)) if chosen_answer_iters else float("nan"),
        "max_chosen_answer_iter": int(np.max(chosen_answer_iters)) if chosen_answer_iters else float("nan"),
        "avg_extra_repetitions": _safe_mean(extra_repetitions),
        "min_extra_repetitions": int(np.min(extra_repetitions)) if extra_repetitions else float("nan"),
        "max_extra_repetitions": int(np.max(extra_repetitions)) if extra_repetitions else float("nan"),
        "valid_questions": len(chosen_answer_iters),
    }
