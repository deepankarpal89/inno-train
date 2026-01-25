from numpy.testing import print_assert_equal
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv
from scripts.storage_client import StorageClient
import tempfile
import logging
from typing import Dict, Any, Optional

load_dotenv()


def read_yaml(file_path: str):
    """Read a YAML file and return the contents as a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ProjectYamlBuilder:
    def __init__(self, logger=None) -> None:
        self.yaml_data = {}
        self.task_config = {}
        self.default_config = {}
        self.base_config_dir = (
            Path(os.path.dirname(os.path.dirname(__file__))) / "config"
        )
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        # Load default configuration
        self._load_default_config()

    def _load_default_config(self) -> None:
        """
        Load default configuration from default_config.yaml.
        """
        default_config_file = self.base_config_dir / "default_config.yaml"
        if not default_config_file.exists():
            self.logger.warning(f"Default config file not found: {default_config_file}")
            return

        try:
            with open(default_config_file, "r", encoding="utf-8") as f:
                self.default_config = yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Error loading default config: {e}")
            self.default_config = {}

    def _load_task_config(self, task_type: str) -> Dict[str, Any]:
        """
        Load task-specific configuration from YAML file.

        Args:
            task_type: Type of task (text_classification, text_extraction, etc.)

        Returns:
            Dictionary containing task-specific configuration
        """
        config_file = self.base_config_dir / f"{task_type}.yaml"
        task_config = {}

        if not config_file.exists():
            self.logger.warning(f"Task config file not found: {config_file}")
        else:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    task_config = yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error(f"Error loading task config: {e}")

        # Merge configurations: default config -> task-specific
        merged_config = {}

        # Add default configuration
        merged_config.update(self.default_config)

        # Add task-specific overrides
        merged_config.update(task_config)

        return merged_config

    def _add_project(self, data):
        if not data or "project" not in data or not data["project"]:
            raise ValueError("Invalid data: 'project' key is missing or empty")

        project = data["project"]
        self.yaml_data["project_name"] = project.get("name", "")
        self.yaml_data["project_id"] = project.get("id", "")
        self.yaml_data["project_description"] = project.get("description", "")
        self.yaml_data["project_task_type"] = project.get(
            "task_type", "text_classification"
        )
        self.yaml_data["training_run_id"] = data.get("training_run_id", "")
        self.yaml_data["training_s3_path"] = (
            f"media/projects/{self.yaml_data['project_id']}/{self.yaml_data['training_run_id']}"
        )
        self.yaml_data["config_s3_path"] = (
            f"{self.yaml_data['training_s3_path']}/config.yaml"
        )
        self.yaml_data["config_file_path"] = f"projects_yaml/config.yaml"

    def _add_prompt(self, data):
        if not data or "prompt" not in data or not data["prompt"]:
            raise ValueError("Invalid data: 'prompt' key is missing or empty")

        prompt = data["prompt"]
        self.yaml_data["prompt_id"] = prompt.get("id", "")
        self.yaml_data["prompt_name"] = prompt.get("name", "")
        self.yaml_data["system_prompt"] = prompt.get("content", "")
        self.yaml_data["think_flag"] = prompt.get("think_flag", True)
        self.yaml_data["seed_text"] = prompt.get("seed_text", "")

    def _add_train_dataset(self, data):
        if not data or "train_dataset" not in data or not data["train_dataset"]:
            raise ValueError("Invalid data: 'train_dataset' key is missing or empty")

        dataset = data["train_dataset"]
        self.yaml_data["train_dataset_id"] = dataset.get("id", "")
        self.yaml_data["train_s3_path"] = f'media/{dataset.get("file_name", "")}'
        self.yaml_data["train_file_name"] = os.path.basename(
            self.yaml_data["train_s3_path"]
        )
        self.yaml_data["train_file_path"] = f"data/{self.yaml_data['train_file_name']}"

    def _add_eval_dataset(self, data):
        if not data or "eval_dataset" not in data or not data["eval_dataset"]:
            raise ValueError("Invalid data: 'eval_dataset' key is missing or empty")

        dataset = data["eval_dataset"]
        self.yaml_data["eval_dataset_id"] = dataset.get("id", "")
        self.yaml_data["eval_s3_path"] = f'media/{dataset.get("file_name", "")}'
        self.yaml_data["eval_file_name"] = os.path.basename(
            self.yaml_data["eval_s3_path"]
        )
        self.yaml_data["eval_file_path"] = f"data/{self.yaml_data['eval_file_name']}"
        self.yaml_data["cv_file_path"] = self.yaml_data["eval_file_path"]

    def _add_reward(self):
        self.yaml_data["reward"] = self.task_config.get("reward", "classify_text")
        self.yaml_data["reward_params"] = self.task_config.get("reward_params", {})

    def _add_ref_model(self):
        self.yaml_data["ref_model_name"] = self.task_config.get(
            "ref_model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        self.yaml_data["ref_model_params"] = self.task_config.get(
            "ref_model_params", {}
        )

    def _add_model(self):
        self.yaml_data["model_name"] = self.task_config.get(
            "model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        )
        self.yaml_data["max_new_tokens"] = self.task_config.get("max_new_tokens", 250)
        self.yaml_data["model_params"] = self.task_config.get("model_params", {})

    def _add_trajectory(self):
        self.yaml_data["trajectory_name"] = self.task_config.get(
            "trajectory_name", "run_1"
        )
        self.yaml_data["trajectory_count"] = self.task_config.get("trajectory_count", 2)
        self.yaml_data["trajectory_params"] = self.task_config.get(
            "trajectory_params", {}
        )

    def _add_train_params(self):
        self.yaml_data["start_iteration"] = self.task_config.get("start_iteration", 1)
        self.yaml_data["topk"] = self.task_config.get("topk", 1)
        self.yaml_data["save_every_epoch"] = self.task_config.get("save_every_epoch", 1)
        self.yaml_data["accumulation_steps"] = self.task_config.get(
            "accumulation_steps", 1
        )
        self.yaml_data["no_epochs"] = self.task_config.get("no_epochs", 1)
        self.yaml_data["no_iterations"] = self.task_config.get("no_iterations", 2)
        self.yaml_data["loss"] = self.task_config.get("loss", "gspo")
        self.yaml_data["model_epoch"] = self.task_config.get("model_epoch", 1)

    def _add_eval_params(self):
        self.yaml_data["topk_eval"] = self.task_config.get("topk_eval", 1)

    def write_yaml(self, data: dict, file_path: str):
        """Write a dictionary to a YAML file at the given location."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def get_yaml_dict(self):
        return self.yaml_data

    def _add_data(self, data):
        self._add_project(data)
        self._add_prompt(data)
        self._add_train_dataset(data)
        self._add_eval_dataset(data)

        # Load task-specific configuration
        task_type = self.yaml_data.get("project_task_type", "text_classification")
        self.task_config = self._load_task_config(task_type)

        # Add task-specific configurations
        self._add_reward()
        self._add_ref_model()
        self._add_model()
        self._add_trajectory()
        self._add_train_params()
        self._add_eval_params()

    def create_yaml(self, data, file_path="config.yaml"):
        self._add_data(data)
        self.write_yaml(self.yaml_data, file_path)

    def save_to_s3(self) -> bool:
        """
        Save the YAML data directly to AWS S3 storage.

        Args:
            bucket_name: Name of the bucket in AWS S3
            object_name: Object name (key) to save as in AWS S3

        Returns:
            bool: True if upload was successful, False otherwise
        """
        bucket_name = os.getenv("BUCKET_NAME")
        object_name = self.yaml_data["config_s3_path"]
        try:

            # Create a temporary file to store the YAML
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as temp_file:
                # Write YAML data to temp file
                import yaml

                yaml.safe_dump(self.yaml_data, temp_file, sort_keys=False)
                temp_file_path = temp_file.name

            try:
                # Initialize storage client and upload
                storage = StorageClient()
                success = storage.upload_file(
                    bucket_name=bucket_name,
                    object_name=object_name,
                    file_path=temp_file_path,
                    content_type="application/yaml",
                )

                if success:
                    self.logger.info(
                        f"Successfully uploaded YAML to S3: {bucket_name}/{object_name}"
                    )
                else:
                    self.logger.info(f"Failed to upload YAML to S3")

                return success

            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    self.logger.info(f"Warning: Failed to delete temporary file: {e}")

        except Exception as e:
            self.logger.info(f"Error saving YAML to S3: {e}")
            return False


def main():
    data = {
        "training_run_id": "7b1be4c4-084d-46d7-948d-12b04b26b049",
        "project": {
            "id": "7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd",
            "name": "spam local",
            "description": "testing spam on local",
            "task_type": "text_classification",
        },
        "prompt": {
            "id": "c2e62aa3-cb7f-43e1-ba57-5817909ef04a",
            "name": "short prompt",
            "content": "You are a helpful assistant that classifies text messages as spam or not spam. For each message, respond with 'spam' or 'not spam'",
        },
        "train_dataset": {
            "id": "996c0774-5676-41d9-a94b-b5b78d694828",
            "name": "spam train",
            "file_url": "http://localhost:9000/innotone-media/media/projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/train/spam_train_test.csv",
            "file_name": "projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/train/spam_train_test.csv",
            "file_format": "csv",
            "text_col": "text",
            "label_col": "label",
            "think_col": "",
        },
        "eval_dataset": {
            "id": "a11fe5b8-639a-4486-b215-f3e1b4e29216",
            "name": "spam test",
            "file_url": "http://localhost:9000/innotone-media/media/projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/eval/spam_cv_test.csv",
            "file_name": "projects/7bce834a-bd56-4fa6-89d6-dcd2acb0b4cd/eval/spam_cv_test.csv",
            "file_format": "csv",
            "text_col": "text",
            "label_col": "label",
            "think_col": "",
        },
        "metadata": {},
    }

    pyb = ProjectYamlBuilder()
    pyb.create_yaml(data, "config.yaml")
    # pyb.save_to_s3()
    print(pyb.get_yaml_dict())


if __name__ == "__main__":
    main()
