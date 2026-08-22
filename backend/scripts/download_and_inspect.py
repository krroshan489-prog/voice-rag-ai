import os
import requests
import pandas as pd
import json

url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"
save_path = os.path.join(os.path.dirname(__file__), "../data/hinval.parquet")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

print(f"Downloading MSMARCO-XI validation sample from {url}...")
resp = requests.get(url, timeout=30)
if resp.status_code == 200:
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print(f"Downloaded successfully! File size: {len(resp.content)} bytes.")

    df = pd.read_parquet(save_path)
    print(f"\n================ MSMARCO-XI SCHEMA DETAILS ================")
    print(f"Dataset Shape: {df.shape} (Rows x Columns)")
    print("Columns:", list(df.columns))
    print("\nSAMPLE RECORD 0:")
    sample = df.iloc[0].to_dict()
    for col, val in sample.items():
        v_str = str(val)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  • {col} ({type(val).__name__}): {v_str}")
    print("===========================================================\n")
else:
    print(f"Failed to download parquet. Status: {resp.status_code}")
