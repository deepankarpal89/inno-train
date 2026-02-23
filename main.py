import yaml
from pathlib import Path

file_path = "/Users/deepankarpal/Projects/innotone/inno-train/test_requests/image_extraction_request_20260213_185645_7b4757cf-c8fb-41d4-b841-fbbf6ae8359e.yaml"
path = Path(file_path)
    
if not path.exists():
    raise FileNotFoundError(f"YAML file not found: {file_path}")
    
with open(path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config['project_name'],config['train_output_cols'])