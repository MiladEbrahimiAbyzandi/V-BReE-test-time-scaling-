# V-BReE

V-BReE stands for **Variance-thresholded Blinded Refinement Ensemble**, a multi-agent reasoning workflow for multiple-choice question answering. The project runs several language models in sequence, lets each model critique and refine the previous answer, and stops when the moving variance of model scores falls below a threshold.

This repository currently focuses on experiments with **MMLU-Pro** and includes:

- provider wrappers for Hugging Face Inference and OpenAI
- the ensemble orchestration logic
- dataset loading and sampling helpers
- evaluation utilities for accuracy, reasoning quality, confidence, and efficiency
- runnable scripts for smoke tests and full experiments

## How It Works

For each question, V-BReE:

1. selects a starting model
2. asks that model to answer the question and score the previous answer
3. passes the updated answer to the next model as context, without treating it as ground truth
4. tracks a moving average and moving variance of model scores over the latest model window
5. chooses the answer from the lowest-variance window as the final response

The core implementation lives in `src/vbree/orchestration/ensemble.py`.

## Project Structure

```text
.
+-- configs/
+-- notebooks/
+-- runs/
+-- scripts/
|   +-- run_experiment.py
|   +-- run_smoke_test.py
|   +-- run_vbree_testing.py
|   +-- sample_dataset.py
|   +-- test_provider.py
+-- src/vbree/
|   +-- data/
|   +-- evaluation/
|   +-- orchestration/
|   +-- prompts/
|   +-- providers/
|   +-- utils/
+-- pyproject.toml
+-- requirements.txt
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The package targets **Python 3.10+**.

## Environment Variables

Create a `.env` file in the project root with the API keys you want to use:

```env
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

Notes:

- `HfProvider` reads `HF_TOKEN`
- `OpenAiProvider` reads `OPENAI_API_KEY`
- the default experiment script uses Hugging Face Inference providers
- the reasoning judge in `scripts/run_experiment.py` also uses a Hugging Face-backed provider

## Data

By default, experiments load **`TIGER-Lab/MMLU-Pro`** from Hugging Face.

You can also pass a local CSV. The loader expects at least these columns:

- `question_id`
- `question`
- `options`
- `category`
- `answer`
- `cot_content` if you want to run reasoning-quality evaluation

Important:

- `options` must be a Python-list-like column, for example `["A", "B", "C", "D"]`
- if `options` is stored as a string, the loader will try to parse it with `literal_eval`

## Quick Start

Run the package smoke test:

```bash
python scripts/run_smoke_test.py
```

Check that a provider can generate text:

```bash
python scripts/test_provider.py
```

Create a balanced sample from MMLU-Pro and save it to `runs/mmlu_pro_sample.csv`:

```bash
python scripts/sample_dataset.py
```

Run the main experiment:

```bash
python scripts/run_experiment.py --run-name demo
```

Run the main experiment with a local CSV:

```bash
python scripts/run_experiment.py --run-name sample --data-csv runs/mmlu_pro_sample.csv
```

## What `run_experiment.py` Does

The main script:

- loads environment variables with `python-dotenv`
- loads MMLU-Pro from Hugging Face or a local CSV
- initializes `Qwen/Qwen2.5-7B-Instruct:together`
- initializes `meta-llama/Llama-3.1-8B-Instruct:cerebras`
- initializes `google/gemma-2-9b-it` via `featherless-ai`
- runs the ensemble in batches of 50 questions
- saves per-batch results and a combined `results.csv`
- computes accuracy
- computes reasoning improvement
- computes confidence analysis
- computes efficiency metrics
- writes a final `summary.json`

## Outputs

Each run is saved under `runs/<run-name>/`.

Typical outputs:

- `batch_1_results.csv`, `batch_2_results.csv`, ...
- `results.csv`
- `summary.json`

The result rows include fields such as:

- `id`
- `iteration`
- `model`
- `previous_answer`
- `updated_answer`
- `selected_choice`
- `score`
- `score_moving_avg`
- `score_moving_variance`
- `chosen_response`
- `confidence_score`

## Using the Ensemble in Code

```python
from vbree.orchestration.ensemble import Ensemble
from vbree.providers.hf_provider import HfProvider
from vbree.data.mmlu_pro import load_mmlu_pro

providers = {
    "Qwen/Qwen2.5-7B-Instruct:together": HfProvider("Qwen/Qwen2.5-7B-Instruct:together"),
    "meta-llama/Llama-3.1-8B-Instruct:cerebras": HfProvider("meta-llama/Llama-3.1-8B-Instruct:cerebras"),
}

ensemble = Ensemble(providers=providers, verbose=True)
ensemble.add_model("Qwen/Qwen2.5-7B-Instruct:together")
ensemble.add_model("meta-llama/Llama-3.1-8B-Instruct:cerebras")

data = load_mmlu_pro(split="validation")

results = ensemble.run(
    data=data.head(5),
    id_col="question_id",
    question_col="question",
    choices_col="options",
    domain_col="category",
)
```

## Evaluation Metrics

The evaluation utilities in `src/vbree/evaluation/metrics.py` provide:

- `accuracy_score`: compares the chosen final answer with ground truth
- `reasoning_analysis`: asks a judge model to compare early and final reasoning against reference CoT
- `confidence_analysis`: compares confidence for correct vs. incorrect answers
- `efficiency`: summarizes how many refinement iterations were needed

## Current Limitations

- the default scripts are tuned specifically for MMLU-Pro column names
- `reasoning_analysis` requires a reference reasoning column such as `cot_content`
- remote inference can fail because of rate limits, provider issues, or networking instability
- the current experiment configuration is defined directly in scripts rather than a CLI config system

## Version

Current package version: `0.1.0`
