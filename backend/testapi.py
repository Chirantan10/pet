import requests

url = "http://127.0.0.1:5000//predict"
files = {"file": open("testwhin.wav", "rb")}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open("response_audio.mp3", "wb") as f:
        f.write(response.content)
    print("Audio response saved as response_audio.mp3")
else:
    print("Error:", response.text)
