from scripts.project_yaml_builder import ProjectYamlBuilder


def test_task_config(task_type):
    data = {
        "project": {
            "id": "test-id",
            "name": "test",
            "description": "test",
            "task_type": task_type,
        },
        "prompt": {"id": "prompt-id", "name": "prompt", "content": "test"},
        "train_dataset": {"id": "train-id", "file_name": "train.csv"},
        "eval_dataset": {"id": "eval-id", "file_name": "eval.csv"},
    }

    builder = ProjectYamlBuilder()
    builder.create_yaml(data, f"test/output/test_{task_type}.yaml")
    print(f"Generated YAML for {task_type}:")

    with open(f"test_{task_type}.yaml", "r") as f:
        print(f.read())
    print("-" * 80)


if __name__ == "__main__":
    # Test with different task types
    test_task_config("text_classification")
    # test_task_config("text_extraction")
    # test_task_config("image_classification")
    # test_task_config("image_extraction")
