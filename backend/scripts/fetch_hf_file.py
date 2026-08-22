import os
import json
import warnings
warnings.filterwarnings("ignore")

try:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print("Fetching validation/hinval.parquet using hf_hub_download...", flush=True)
    file_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI",
        filename="validation/hinval.parquet",
        repo_type="dataset"
    )
    print(f"Downloaded file location: {file_path}", flush=True)

    df = pd.read_parquet(file_path)
    print("\n================ MSMARCO-XI DISCOVERED SCHEMA ================", flush=True)
    print(f"Split: validation (hinval.parquet) | Shape: {df.shape}", flush=True)
    print("Available Columns:", list(df.columns), flush=True)
    print("\nSAMPLE RECORD 0:")
    rec = df.iloc[0].to_dict()
    for k, v in rec.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  • {k} ({type(v).__name__}): {v_str}", flush=True)
    print("==============================================================\n", flush=True)

except Exception as e:
    print("Error:", e, flush=True)
