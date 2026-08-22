import sys
import json
import warnings
warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
    print("Loading dataset in streaming mode...", flush=True)
    ds = load_dataset("ai4bharat/MSMARCO-XI", data_files={"validation": "validation/hinval.parquet"}, streaming=True)
    sample = next(iter(ds["validation"]))
    print("SUCCESS! Columns found:", list(sample.keys()), flush=True)
    print("\nFIELDS AND SAMPLE VALUES:")
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  • {k} ({type(v).__name__}): {v_str}", flush=True)
except Exception as e:
    print("Error:", e, flush=True)
