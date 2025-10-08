# lambda_cloud/client.py
import requests
from requests.auth import HTTPBasicAuth
import os
from typing import List, Optional
from dotenv import load_dotenv
import time

class LambdaClient:
    """A client for interacting with the Lambda Cloud API."""
    
    BASE_URL = "https://cloud.lambda.ai/api/v1"
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the Lambda Cloud client."""
        load_dotenv()
        if api_key is None:
            
            api_key = os.getenv('LAMBDA_API_KEY')
            if not api_key:
                raise ValueError("No API key provided and LAMBDA_API_KEY environment variable not set")
        
        self.ssh_key_name = os.getenv('SSH_KEY_NAME')
        
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(api_key, '')
        self.base_url = base_url or self.BASE_URL
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an HTTP request to the Lambda Cloud API."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise
    
    def list_available_instances(self):
        data = self._make_request('GET','/instance-types')['data']
        avail_gpus = {k:v for k,v in data.items() if len(v['regions_with_capacity_available']) > 0 and v['instance_type']['specs']['gpus'] == 1}
        selected_gpu = sorted(avail_gpus.items(), key=lambda x: x[1]['instance_type']['price_cents_per_hour'])[0][0]
        selected_region = avail_gpus[selected_gpu]['regions_with_capacity_available'][0]['name']
        resp = {'name':selected_gpu,'region':selected_region}
        return resp

    def list_instances(self):
        """List all Lambda Cloud instances."""
        resp = self._make_request('GET', '/instances')
        return resp['data']
    
    def get_instance(self, instance_id: str):
        """Get details of a specific instance."""
        resp = self._make_request('GET', f'/instances/{instance_id}')
        return resp['data'] if resp.get('data') else None
    
    def get_instance_status(self,instance_id: str):
        """Get the status of a specific instance."""
        resp = self.get_instance(instance_id)
        return resp['status'] if resp else None

    def terminate_instance(self, instance_id: str):
        """Terminate a specific instance."""
        try:
            self._make_request('POST', '/instance-operations/terminate',json={'instance_ids': [instance_id]})
            return True
        except Exception as e:
            print(f"Failed to terminate instance {instance_id}: {e}")
            return False
    
    def list_ssh_keys(self):
        """List all available SSH keys."""
        resp = self._make_request('GET', '/ssh-keys')
        return resp['data']
    
    def launch_instance(self,instance_type_name: str,region_name: str,ssh_key_names: List[str] = None,name: str = None,file_system_names: List[str] = None,
        **kwargs):
        """Launch a new instance."""
        payload = {
            'instance_type_name': instance_type_name,
            'region_name': region_name,
            'ssh_key_names': ssh_key_names or [self.ssh_key_name],
            **kwargs
        }
        
        if name:
            payload['name'] = name
        if file_system_names:
            payload['file_system_names'] = file_system_names
        
        try:
            resp = self._make_request('POST', '/instance-operations/launch', json=payload)
            instance_id = resp['data']['instance_ids'][0]

            while self.get_instance_status(instance_id) != 'active':
                time.sleep(60)
                print(f"Waiting for instance {instance_id} to be active...")
            
            print(f"Instance {instance_id} is active.")
                
            return self.get_instance(instance_id)
        except Exception as e:
            print(f"Failed to launch instance: {e}")
            return None

    def list_file_systems(self):
        """List all available file systems."""
        resp = self._make_request('GET', '/file-systems')
        return resp['data'] if resp else None

    def get_instance_file_systems(self, instance_id: str):
        """Get file systems attached to a specific instance."""
        instance = self.get_instance(instance_id)
        if not instance:
            return []
        
        all_file_systems = self.list_file_systems()
        
        return [
            fs for fs in all_file_systems 
            if fs['name'] in (instance['file_system_names'] or [])
        ]

    def get_file_system(self, identifier: str, by_name: bool = False):
        """Get a file system by ID or name."""
        file_systems = self.list_file_systems()
        if by_name:
            for fs in file_systems:
                if fs['name'] == identifier:
                    return fs
        else:
            for fs in file_systems:
                if fs['id'] == identifier:
                    return fs
        return None