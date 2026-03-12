import pandas as pd
import numpy as np
from datasets import load_dataset
from ast import literal_eval
from typing import Sequence

def _coerce_options_to_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = literal_eval(value)
            if isinstance(parsed, (list,tuple)):
                return list(parsed)
        except Exception:
            raise ValueError(f"Cannot parse options string: {value}")
    raise ValueError(f"Unsupported options format: {value} (type {type(value)})")

def _normalize_domains(domain: str | Sequence[str] | None) -> list[str] | None:
    if domain is None:
        return None
    if isinstance(domain, str):
        return [domain]

    domains = list(domain)
    if not domains:
        raise ValueError("domain must not be empty.")
    return domains

def load_mmlu_pro(
    split: str = "validation",
    sample: bool = False,
    domain: str | Sequence[str] | None = None,
    n_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:

    df = load_dataset("TIGER-Lab/MMLU-Pro", split=split).to_pandas()
    df["options"] = df["options"].apply(_coerce_options_to_list)

    domains = _normalize_domains(domain)
    if domains is not None:
        df = df[df["category"].isin(domains)]
        if df.empty:
            raise ValueError(f"No rows found for domain(s): {domains}")

    if sample:
        if n_samples is None:
            raise ValueError("n_samples must be provided when sample=True.")
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0.")
        sample_part = []
        for cat in df["category"].unique():
            cat_count = (df["category"] == cat).sum()
            sample_size = min(n_samples, cat_count)
            sample_part.append(df[df["category"] == cat].sample(n=sample_size, random_state=random_state))
        df = pd.concat(sample_part, ignore_index=True)

    return df.reset_index(drop=True)
