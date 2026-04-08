import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN not set")
    exit(1)

url = "https://router.huggingface.co/v1/models"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    models = response.json()
    # Print first 20 models
    for m in models.get("data", [])[:30]:
        print(m.get("id"))
except Exception as e:
    print(f"Error fetching models: {e}")
