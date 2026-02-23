import json
import os
from pathlib import Path
from project_yaml_builder import ProjectYamlBuilder


def process_request_file(request_file_path: str, output_dir: str):
    """
    Process a single request JSON file and generate corresponding YAML.

    Args:
        request_file_path: Path to the request JSON file
        output_dir: Directory to save the generated YAML
    """
    with open(request_file_path, "r") as f:
        request_data = json.load(f)

    if not request_data.get("success") or "data" not in request_data:
        print(f"Skipping {request_file_path}: Invalid request structure")
        return

    data = request_data["data"]["request_data"]

    task_type = data["project"]["task_type"]
    request_filename = Path(request_file_path).stem

    output_filename = f"{task_type}_{request_filename}.yaml"
    output_path = os.path.join(output_dir, output_filename)

    builder = ProjectYamlBuilder()
    builder.create_yaml(data, output_path)

    print(f"✅ Generated: {output_filename}")
    print(f"   Task Type: {task_type}")
    print(f"   Project: {data['project']['name']}")

    if task_type in ["image_classification", "image_extraction"]:
        train_image_root = data.get("train_dataset", {}).get("image_root", "")
        eval_image_root = data.get("eval_dataset", {}).get("image_root", "")
        if train_image_root or eval_image_root:
            print(f"   📁 Image roots included for transfer")
    print()


def main():
    script_dir = Path(__file__).parent
    test_requests_dir = script_dir.parent / "test_requests"

    if not test_requests_dir.exists():
        print(f"Error: test_requests directory not found at {test_requests_dir}")
        return

    request_files = sorted(test_requests_dir.glob("request_*.json"))

    if not request_files:
        print("No request files found in test_requests directory")
        return

    print(f"Found {len(request_files)} request file(s)\n")
    print("=" * 70)

    for request_file in request_files:
        process_request_file(str(request_file), str(test_requests_dir))

    print("=" * 70)
    print(f"✅ Successfully generated {len(request_files)} YAML file(s)")


if __name__ == "__main__":
    main()
