from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename="ms_marco_translations.py", repo_type="dataset")
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

print("=== MSMARCO-XI DATASET LOADING SCRIPT CONTENTS ===")
print(code[:1500])
