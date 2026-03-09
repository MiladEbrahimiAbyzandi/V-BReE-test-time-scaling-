import pandas as pd
import numpy as np
from datasets import load_dataset
from ast import literal_eval

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

def load_demo_data(split: str = "validation") -> pd.DataFrame:
    
    df = load_dataset("TIGER-Lab/MMLU-Pro", split = split). to_pandas()
    df["options"] = df["options"].apply(_coerce_options_to_list)

    return df