from vbree.data.mmlu_pro import load_mmlu_pro
import pandas as pd
from pathlib import Path

def sample_mmlu_pro(data: pd.DataFrame, n_samples: int = 250, category: str = "category") -> pd.DataFrame:
    if category not in data.columns:
        raise ValueError(f"Category column '{category}' not found in data.")
    sample_part = []

    for category, group in data.groupby(category):
        sample_size = min(n_samples, len(group))
        sample_part.append(group.sample(sample_size, random_state=42))

    sampled_data = pd.concat(sample_part, ignore_index=True)

    return sampled_data

def main():
    data = load_mmlu_pro(split = "test")
    sampled_data = sample_mmlu_pro(data, n_samples=250, category="category")
    output_path = Path(__file__).resolve().parent.parent / "runs" / "mmlu_pro_sample.csv"
    sampled_data.to_csv(output_path, index=False)

    print("\nSaved sampled dataset to:")
    print(output_path)

if __name__ == "__main__":
    main()
