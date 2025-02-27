import requests

url = "http://localhost:8000/upload_images/"
files = {"file": open(
    "/home/wz/AI/ai_tookit_ort/resources/test_lite_age_googlenet.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
