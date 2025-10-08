from .lambda_client import LambdaClient
from .ssh_executor import SshExecutor

class TrainingHandler:
    def __init__(self):
        self.lambda_client = LambdaClient()
        self.ssh_executor = SshExecutor()

    