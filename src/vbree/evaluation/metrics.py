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
        raise ValueError(f"Ground truth DataFrame must contain columns '{dataset_id_col}' and '{ground_truth_col}'.")

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
    return f"""You are an expert evaluator assessing mathematical reasoning quality.

Question: {question}

Reference Reasoning (Ground Truth):
{cot_ground_truth}

Answer A (Starting Response):
{start_answer}

Answer B (Final Response):
{end_answer}

Task:
Compare BOTH answers against the ground truth reasoning.
Focus on logical validity, not word matching.
Different wording is fine as long as logic is sound.

Scoring guide:
- 0-40: incorrect logic or wrong conclusion
- 41-70: partially correct logic, minor gaps
- 71-100: logically sound reasoning, correct conclusion

Important rules:
- Score each answer INDEPENDENTLY against ground truth
- Do not reward B just because it is longer
- Focus on: are logical steps correct? is conclusion valid?

Return ONLY valid JSON:
{{
    "score_a": <number 0-100>,
    "score_b": <number 0-100>,
    "feedback_a": "<one sentence>",
    "feedback_b": "<one sentence>",
    "improved": <true if score_b > score_a, else false>
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
    if n_models <= 0:
        raise ValueError("n_models must be >= 1.")

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

        try:
            result = json.loads(raw)
            score_a = float(result["score_a"])
            score_b = float(result["score_b"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
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

        chosen_rows = group[group["chosen_response"] == True]  # noqa: E712
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
