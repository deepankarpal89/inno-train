from scripts.s3_to_server_transfer import S3ToServerTransfer
import os
import logging
import time

logger = logging.getLogger(__name__)
start = time.time()
file_transfer = S3ToServerTransfer(logger=logger)
server_ip = '129.213.147.80'
s3_bucket = os.getenv("BUCKET_NAME")
server_folder_path = 'output/spam local/training/run_2/optimizer/optimizer_epoch_1.pt'
s3_folder_path = 'media/projects/c1aa9a60-0944-4261-aaea-517dafcd74bc/spam local/training/run_2/optimizer/optimizer_epoch_1.pt'
file_transfer.transfer_files_to_s3(server_folder_path=server_folder_path,s3_folder_path=s3_folder_path,s3_bucket=s3_bucket,server_ip=server_ip)
end = time.time()
print(f"Time taken: {end - start}")
