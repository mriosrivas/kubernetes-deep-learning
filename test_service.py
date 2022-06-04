import requests
import json

data = {"url": "https://bit.ly/3N3qWuF"}
 
url = 'http://localhost:8080/predict'

response = requests.post(url, json=data)
result = response.json()

print(json.dumps(result, indent=2))
