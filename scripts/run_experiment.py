from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import ast
from vbree.orchestration.ensemble import Ensemble
from vbree.providers.hf_provider import HfProvider
from vbree.evaluation.metrics import accuracy_score, reasoning_analysis, confidence_analysis, efficiency
from vbree.data.mmlu_pro import load_mmlu_pro

def build_run_dir (run_name: str) -> Path:
    run_dir = Path(__file__).resolve().parent.parent/ "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def main():

    load_dotenv()

    run_name = "demo_step6"
    run_dir = build_run_dir(run_name)

    data = load_mmlu_pro()

    providers = {
        "Qwen/Qwen2.5-7B-Instruct:together": HfProvider("Qwen/Qwen2.5-7B-Instruct:together"),
        "meta-llama/Llama-3.1-8B-Instruct:cerebras": HfProvider("meta-llama/Llama-3.1-8B-Instruct:cerebras"),
        "mistralai/Mistral-7B-Instruct-v0.2": HfProvider("mistralai/Mistral-7B-Instruct-v0.2", provider="featherless-ai")
    }

    ens  = Ensemble(providers=providers, verbose=True)

    ens .add_model("Qwen/Qwen2.5-7B-Instruct:together")
    ens .add_model("meta-llama/Llama-3.1-8B-Instruct:cerebras")
    ens .add_model("mistralai/Mistral-7B-Instruct-v0.2")

    results = ens .run(
        data=data,
        id_col="question_id",
        question_col="question",
        choices_col="options", 
        domain_col="category",
        temperature=0.0,
    )

    results_path = run_dir / "results.csv"
    results.to_csv(results_path, index=False)

    results_path = run_dir / "results.csv"
    results.to_csv(results_path, index=False)

    # ---------------------------
    # 6) Run metrics
    # ---------------------------
    acc = accuracy_score(
        results=results,
        ground_truth=data,
        dataset_id_col="question_id",
        ground_truth_col="answer"
    )

    judge_provider = HfProvider("moonshotai/Kimi-K2-Instruct-0905:groq")

    reasoning = reasoning_analysis(
        results=results,
        ground_truth=data,
        judge_provider=judge_provider,
        n_models=len(ens.models),
        cot_content="cot_content",  
        dataset_id_col="question_id",
        verbose=True
    )

    confidence = confidence_analysis(
        results=results,
        ground_truth=data,
        dataset_id_col="question_id",
        ground_truth_col="answer"

    )
    eff = efficiency(results)

    summary = {
        "run_name": run_name,
        "n_models": len(ens.models),
        "models": ens.models,
        "accuracy": acc,
        "reasoning": reasoning,
        "confidence": confidence,
        "efficiency": eff,
    }

    # ---------------------------
    # 7) Save summary
    # ---------------------------
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nExperiment completed.")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

