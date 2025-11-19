from scripts.lambda_client import LambdaClient
from scripts.project_yaml_builder import ProjectYamlBuilder
from scripts.s3_to_server_transfer import S3ToServerTransfer
from scripts.ssh_executor import SshExecutor
from dotenv import load_dotenv
import os

import time
from datetime import timedelta

# At the very beginning of the script
start_time = time.time()

load_dotenv()

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
pyb.create_yaml(data)
pyb.save_to_s3()

gpu_client = LambdaClient()
gpu_config = gpu_client.list_available_instances()

if not gpu_config:
    raise Exception("No available GPU instances")

instance_type_name = gpu_config['name']
instance_region_name = gpu_config['region']
instance_config = gpu_client.launch_instance(instance_type_name, instance_region_name)

if not instance_config:
    raise Exception("Failed to launch GPU instance")

instance_id = instance_config['id']
instance_ip = instance_config['ip']

file_transfer = S3ToServerTransfer()
server_ip = instance_ip

#transfer train
s3_bucket = os.getenv("BUCKET_NAME")
s3_prefix = pyb.yaml_data['train_s3_path']
server_ip = instance_ip
server_path = pyb.yaml_data['train_file_path']
file_transfer.transfer_file_to_server(s3_bucket, s3_prefix, server_ip, server_path)

#transfer eval
s3_prefix = pyb.yaml_data['eval_s3_path']
server_path = pyb.yaml_data['eval_file_path']
file_transfer.transfer_file_to_server(s3_bucket, s3_prefix, server_ip, server_path)

#transfer config
s3_prefix = pyb.yaml_data['config_s3_path']
server_path = pyb.yaml_data['config_file_path']
print('s3_prefix', s3_prefix)
print('server_path', server_path)
file_transfer.transfer_file_to_server(s3_bucket, s3_prefix, server_ip, server_path)

#run training
se = SshExecutor(ip=server_ip, username='ubuntu')
script_path = 'run_docker_job.sh'
se.upload_file(script_path)
script_name = 'run_docker_job.sh'
se.execute_script(script_name)


#transfer training run files from server to s3
s3_path = pyb.yaml_data['training_s3_path']
server_path = 'output/'
file_transfer.transfer_files_to_s3(server_ip, server_path, s3_bucket, s3_path,recursive=True)

#close instance
gpu_client.terminate_instance(instance_id)

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {str(timedelta(seconds=execution_time))} (HH:MM:SS)")