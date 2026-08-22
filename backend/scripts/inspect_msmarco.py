import sys
import json
import warnings
warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
    print("Loading ai4bharat/MSMARCO-XI dataset via datasets.load_dataset...", flush=True)
    # Load dataset split using Hugging Face datasets library
    ds = load_dataset("ai4bharat/MSMARCO-XI", data_files={"validation": "validation/hinval.parquet"})
    print("Discovered dataset structure:", ds, flush=True)
    val_set = ds["validation"]
    print(f"Validation split length: {len(val_set)}", flush=True)
    sample = val_set[0]
    print("Columns:", list(sample.keys()), flush=True)
    print("\nSAMPLE RECORD SCHEMAS:")
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  - {k} ({type(v).__name__}): {v_str}", flush=True)

except Exception as e:
    print("Error:", e, flush=True)
