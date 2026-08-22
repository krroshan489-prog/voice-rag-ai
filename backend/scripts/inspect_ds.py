import sys
import json
import traceback

print("Starting inspection script...", flush=True)

try:
    from datasets import get_dataset_config_names, load_dataset
    print("Calling get_dataset_config_names...", flush=True)
    configs = get_dataset_config_names("ai4bharat/MSMARCO-XI")
    print(f"Discovered Configs ({len(configs)}):", configs[:10], flush=True)

    config_to_use = configs[0] if configs else None
    print(f"Loading dataset with config='{config_to_use}'...", flush=True)
    
    if config_to_use:
        ds = load_dataset("ai4bharat/MSMARCO-XI", config_to_use)
    else:
        ds = load_dataset("ai4bharat/MSMARCO-XI")

    print("Discovered Splits:", list(ds.keys()), flush=True)
    first_split = list(ds.keys())[0]
    split_data = ds[first_split]
    print(f"Split '{first_split}' has {len(split_data)} records.", flush=True)
    
    sample = split_data[0]
    print("Columns:", list(sample.keys()), flush=True)
    print("Sample record fields:", flush=True)
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 150:
            v_str = v_str[:150] + "..."
        print(f"  - {k} ({type(v).__name__}): {v_str}", flush=True)

except Exception as e:
    print("ERROR OCCURRED:", e, flush=True)
    traceback.print_exc(file=sys.stdout)
