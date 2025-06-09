import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "Yapraklarda kahverengi halkalı lekeler oluşmuş"}
response = requests.post(url, json=payload)
print(response.json())