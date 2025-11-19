import requests

hell_world_url = "http://localhost:8001/api/hello"

response = requests.get(hell_world_url)
print(response.json())
