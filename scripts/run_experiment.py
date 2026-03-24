from pathlib import Path
import argparse
from dotenv import load_dotenv
import pandas as pd
import json
import math
from vbree.orchestration.ensemble import Ensemble
from vbree.providers.hf_provider import HfProvider
from vbree.evaluation.metrics import accuracy_score, reasoning_analysis, confidence_analysis, efficiency
from vbree.data.mmlu_pro import load_mmlu_pro

def build_run_dir (run_name: str) -> Path:
    run_dir = Path(__file__).resolve().parent.parent/ "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a V-BReE experiment.")
    parser.add_argument(
        "--run-name",
        default="demo",
        help="Directory name under runs/ where outputs will be saved.",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Optional CSV dataset path. If omitted, load MMLU-Pro from Hugging Face.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    load_dotenv()

    run_name = args.run_name
    run_dir = build_run_dir(run_name)

    data = load_mmlu_pro(csv_path=args.data_csv)

    providers = {
        "Qwen/Qwen2.5-7B-Instruct:together": HfProvider("Qwen/Qwen2.5-7B-Instruct:together"),
        "meta-llama/Llama-3.1-8B-Instruct:cerebras": HfProvider("meta-llama/Llama-3.1-8B-Instruct:cerebras"),
        "google/gemma-2-9b-it": HfProvider("google/gemma-2-9b-it", provider="featherless-ai")
    }

    ens  = Ensemble(providers=providers, verbose=True)

    ens .add_model("Qwen/Qwen2.5-7B-Instruct:together")
    ens .add_model("meta-llama/Llama-3.1-8B-Instruct:cerebras")
    ens .add_model("google/gemma-2-9b-it")

    batch_size  = 50

    itters = math.ceil(len(data) / batch_size)
    results_parts = []
    
    for batch_idx in range(itters):

        batch_result_path = run_dir / f"batch_{batch_idx+1}_results.csv"
        if batch_result_path.exists():
            print(f"Batch {batch_idx+1} results already exist. Skipping...")
            batch_results = pd.read_csv(batch_result_path)
            results_parts.append(batch_results)
            continue
        
        start = batch_idx * batch_size
        end = min(start + batch_size, len(data))
        batch_data = data.iloc[start:end].copy()



        batch_results = ens .run(
            data=batch_data,
            id_col="question_id",
            question_col="question",
            choices_col="options", 
            domain_col="category",
        )


        batch_results.to_csv(batch_result_path, index=False)
        print(f"Batch {batch_idx+1}/{itters} completed. Results saved to: {batch_result_path}")
        results_parts.append(batch_results)

    if not results_parts:
        raise ValueError("No batch results were loaded or generated.")

    results = pd.concat(results_parts, ignore_index=True)
    results_path = run_dir / "results.csv"
    results.to_csv(results_path, index=False)

    # ---------------------------
    # Run metrics
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

